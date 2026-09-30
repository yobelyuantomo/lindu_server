"""Test logika pembanding ``ml_training/ml_selftest.py`` (tanpa MQTT/perangkat)."""

import unittest

from ml.feature_extractor import FEATURE_NAMES, extract_features, features_to_vector
from ml_training import ml_selftest as st


def _row(ts, pga):
    return {
        "sensor_ts": ts, "pga": pga, "sta_lta": 3.0, "freq_hz": 8,
        "accel_x": pga, "accel_y": 0.0, "accel_z": 0.01,
        "temperature": 25.0, "pressure": 1000.0, "gas_raw": 100,
    }


def _burst():
    # 10 Hz, timestamp berpresisi milidetik, ada puncak di tengah.
    pgas = [0.05, 0.10, 0.20, 0.35, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
    return [_row(1790000000.0 + i * 0.1, p) for i, p in enumerate(pgas)]


def _reply(rows, features):
    return {
        "t_first": rows[0]["sensor_ts"], "t_last": rows[-1]["sensor_ts"],
        "n_samples": len(rows), "features": features, "model": "uji",
    }


class TestEvaluate(unittest.TestCase):
    def test_setara_bila_fitur_sama(self):
        rows = _burst()
        vec = features_to_vector(extract_features(rows))
        kode, ringkasan, tabel = st.evaluate(_reply(rows, vec), rows)
        self.assertEqual(kode, st.EXIT_SETARA, ringkasan)
        self.assertEqual(len(tabel), len(FEATURE_NAMES))

    def test_toleransi_float32(self):
        rows = _burst()
        vec = [v * (1 + 1e-5) for v in features_to_vector(extract_features(rows))]
        kode, _, _ = st.evaluate(_reply(rows, vec), rows)
        self.assertEqual(kode, st.EXIT_SETARA)

    def test_satu_fitur_berbeda_terdeteksi_dan_disebut_namanya(self):
        rows = _burst()
        vec = features_to_vector(extract_features(rows))
        idx = FEATURE_NAMES.index("energy_cumulative")
        vec[idx] = 0.0  # gejala klasik timestamp sub-detik hilang
        kode, ringkasan, _ = st.evaluate(_reply(rows, vec), rows)
        self.assertEqual(kode, st.EXIT_BEDA)
        self.assertIn("energy_cumulative", ringkasan)

    def test_jumlah_fitur_berbeda(self):
        rows = _burst()
        kode, _, _ = st.evaluate(_reply(rows, [0.0] * 5), rows)
        self.assertEqual(kode, st.EXIT_BEDA)

    def test_paket_hilang_tidak_disimpulkan(self):
        rows = _burst()
        vec = features_to_vector(extract_features(rows))
        kode, ringkasan, _ = st.evaluate(_reply(rows, vec), rows[:-2])
        self.assertEqual(kode, st.EXIT_TAK_TERSIMPULKAN)
        self.assertIn("tidak sejajar", ringkasan)

    def test_jendela_tenang_diberi_peringatan(self):
        rows = [_row(1790000000.0 + i, 0.01) for i in range(3)]
        vec = features_to_vector(extract_features(rows))
        kode, ringkasan, _ = st.evaluate(_reply(rows, vec), rows)
        self.assertEqual(kode, st.EXIT_SETARA)
        self.assertIn("PERINGATAN", ringkasan)


class TestSelectRows(unittest.TestCase):
    def test_hanya_rentang_waktu(self):
        rows = _burst()
        dipilih = st.select_rows(rows, rows[2]["sensor_ts"], rows[5]["sensor_ts"])
        self.assertEqual(len(dipilih), 4)


if __name__ == "__main__":
    unittest.main()
