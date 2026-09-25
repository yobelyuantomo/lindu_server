"""Uji ml/feature_extractor.py dengan jendela sintetis yang hasilnya diketahui."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.feature_extractor import (  # noqa: E402
    FEATURE_NAMES,
    MIN_SAMPLES_PER_WINDOW,
    RULE_PGA_MIN,
    build_windows,
    extract_features,
    features_to_vector,
    is_analyzable_window,
    rule_based_verdict,
)


def make_row(ts, pga=0.0, sta_lta=1.0, freq_hz=10.0, ax=0.0, ay=0.0, az=0.0,
             temperature=28.0, pressure=1010.0, gas_raw=100):
    return {
        "sensor_ts": ts, "pga": pga, "sta_lta": sta_lta, "freq_hz": freq_hz,
        "accel_x": ax, "accel_y": ay, "accel_z": az,
        "temperature": temperature, "pressure": pressure, "gas_raw": gas_raw,
    }


class TestRuleBasedVerdict(unittest.TestCase):
    def test_lolos_ketiga_lapis(self):
        self.assertTrue(rule_based_verdict(pga=0.5, sta_lta=3.0, freq_hz=5.0))

    def test_gagal_per_lapis(self):
        self.assertFalse(rule_based_verdict(pga=0.05, sta_lta=3.0, freq_hz=5.0))
        self.assertFalse(rule_based_verdict(pga=0.5, sta_lta=1.0, freq_hz=5.0))
        self.assertFalse(rule_based_verdict(pga=0.5, sta_lta=3.0, freq_hz=35.0))

    def test_tepat_di_ambang_dianggap_lolos(self):
        self.assertTrue(rule_based_verdict(pga=0.12, sta_lta=2.0, freq_hz=20.0))

    def test_freq_hz_hilang_tidak_lolos(self):
        """Regresi bug lama: default 0 membuat payload tanpa freq_hz lolos."""
        self.assertFalse(rule_based_verdict(pga=0.5, sta_lta=3.0, freq_hz=None))


class TestAnalyzableWindow(unittest.TestCase):
    def test_jendela_tenang_ditolak(self):
        rows = [make_row(i * 1.0, pga=0.01) for i in range(6)]
        self.assertFalse(is_analyzable_window(rows))

    def test_jendela_terpicu_diterima(self):
        rows = [make_row(i * 0.1, pga=0.01) for i in range(6)]
        rows[3]["pga"] = RULE_PGA_MIN + 0.1
        self.assertTrue(is_analyzable_window(rows))

    def test_terlalu_sedikit_sampel_ditolak(self):
        rows = [make_row(i * 0.1, pga=0.9) for i in range(MIN_SAMPLES_PER_WINDOW - 1)]
        self.assertFalse(is_analyzable_window(rows))


class TestExtractFeatures(unittest.TestCase):
    def test_kunci_sesuai_kontrak(self):
        rows = [make_row(i * 0.1, pga=0.3) for i in range(10)]
        feats = extract_features(rows)
        self.assertEqual(sorted(feats.keys()), sorted(FEATURE_NAMES))
        self.assertEqual(len(features_to_vector(feats)), len(FEATURE_NAMES))

    def test_jendela_kosong_melempar(self):
        with self.assertRaises(ValueError):
            extract_features([])

    def test_nilai_amplitudo(self):
        rows = [make_row(i * 0.1, pga=p) for i, p in enumerate([0.1, 0.2, 0.6, 0.2])]
        feats = extract_features(rows)
        self.assertAlmostEqual(feats["pga_max"], 0.6)
        self.assertAlmostEqual(feats["pga_mean"], 0.275)
        # Puncak di indeks 2 -> 0.2 detik setelah awal jendela
        self.assertAlmostEqual(feats["time_to_peak"], 0.2, places=6)
        # rise_rate = (0.6 - 0.1) / 0.2
        self.assertAlmostEqual(feats["rise_rate"], 2.5, places=4)

    def test_durasi_di_atas_ambang_pakai_integrasi_waktu(self):
        # 4 sampel berjarak 0.1 s; tiga terakhir di atas ambang.
        # Integrasi memakai selisih waktu, jadi 3 interval x 0.1 s = 0.3 s.
        rows = [make_row(i * 0.1, pga=p) for i, p in enumerate([0.01, 0.5, 0.5, 0.5])]
        feats = extract_features(rows)
        self.assertAlmostEqual(feats["duration_above_thresh"], 0.3, places=6)

    def test_energi_tidak_bergantung_laju_sampling(self):
        """Getaran identik yang disampel 1 Hz vs 10 Hz harus berenergi sama.

        Ini penjaga utama terhadap kebocoran laju sampling: firmware mengirim
        10 Hz saat terpicu dan 1 Hz saat tenang, jadi fitur energi tidak boleh
        ikut berubah hanya karena kepadatan sampelnya berbeda.
        """
        durasi = 2.0
        pga = 0.4
        jarang = [make_row(i * 0.5, pga=pga) for i in range(int(durasi / 0.5) + 1)]
        rapat = [make_row(i * 0.05, pga=pga) for i in range(int(durasi / 0.05) + 1)]
        e_jarang = extract_features(jarang)["energy_cumulative"]
        e_rapat = extract_features(rapat)["energy_cumulative"]
        self.assertAlmostEqual(e_jarang, e_rapat, places=6)

    def test_hv_ratio_membedakan_horizontal_dan_vertikal(self):
        horizontal = [make_row(i * 0.1, pga=0.5, ax=1.0, ay=0.0, az=0.05) for i in range(8)]
        vertikal = [make_row(i * 0.1, pga=0.5, ax=0.05, ay=0.0, az=1.0) for i in range(8)]
        self.assertGreater(extract_features(horizontal)["hv_ratio"], 5.0)
        self.assertLess(extract_features(vertikal)["hv_ratio"], 0.5)

    def test_pressure_delta(self):
        rows = [make_row(i * 0.1, pga=0.3, pressure=1000.0 + i) for i in range(5)]
        self.assertAlmostEqual(extract_features(rows)["pressure_delta"], 4.0)

    def test_field_hilang_tidak_membuat_crash(self):
        rows = [{"sensor_ts": i * 0.1, "pga": 0.3} for i in range(6)]
        feats = extract_features(rows)
        self.assertEqual(sorted(feats.keys()), sorted(FEATURE_NAMES))


class TestBuildWindows(unittest.TestCase):
    def test_potong_berdasarkan_durasi_bukan_jumlah_baris(self):
        # 10 Hz selama 6 detik -> 3 jendela @ 2 detik
        rows = [make_row(i * 0.1) for i in range(60)]
        windows = build_windows(rows, window_seconds=2.0)
        self.assertEqual(len(windows), 3)
        self.assertEqual(len(windows[0]), 20)

    def test_laju_rendah_tetap_terpotong_per_durasi(self):
        # 1 Hz selama 6 detik -> tetap 3 jendela, tapi isinya 2 baris saja
        rows = [make_row(i * 1.0) for i in range(6)]
        windows = build_windows(rows, window_seconds=2.0)
        self.assertEqual(len(windows), 3)
        self.assertEqual(len(windows[0]), 2)

    def test_deret_kosong(self):
        self.assertEqual(build_windows([]), [])


if __name__ == "__main__":
    unittest.main()
