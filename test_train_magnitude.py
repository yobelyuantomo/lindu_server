"""Uji ml_training/train_magnitude.py (estimasi dini puncak guncangan)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.train_magnitude import (  # noqa: E402
    build_events,
    evaluate_baselines,
    formula_magnitudo,
    mae,
    rmse,
    split_events,
)


def baris(ts, pga, node_id="NODE_A", session_id="S001", label="earthquake"):
    return {
        "node_id": node_id, "session_id": session_id, "label": label,
        "sensor_ts": ts, "pga": pga, "sta_lta": 3.0, "freq_hz": 8.0,
        "accel_x": 0.3, "accel_y": 0.2, "accel_z": 0.1,
        "temperature": 28.0, "pressure": 1010.0, "gas_raw": 100.0,
    }


def kejadian(session_id, awal_pga=0.3, puncak_pga=0.9, node_id="NODE_A"):
    """1 detik awal pada awal_pga, lalu memuncak di puncak_pga."""
    rows = [baris(i * 0.1, awal_pga, node_id, session_id) for i in range(10)]
    rows += [baris(1.0 + i * 0.1, puncak_pga, node_id, session_id) for i in range(10)]
    return rows


class TestFormulaMagnitudo(unittest.TestCase):
    def test_pada_pga_referensi(self):
        # log10(0.1/0.1) = 0 -> tepat 5.0
        self.assertAlmostEqual(formula_magnitudo(0.1), 5.0)

    def test_dibatasi_atas_dan_bawah(self):
        self.assertEqual(formula_magnitudo(0.0), 3.0)
        self.assertEqual(formula_magnitudo(10000.0), 9.5)

    def test_monoton_naik(self):
        nilai = [formula_magnitudo(p) for p in (0.15, 0.3, 0.6, 1.2)]
        self.assertEqual(nilai, sorted(nilai))


class TestMetrik(unittest.TestCase):
    def test_mae_dan_rmse(self):
        self.assertAlmostEqual(mae([1.0, 2.0], [1.5, 2.5]), 0.5)
        self.assertAlmostEqual(rmse([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertAlmostEqual(rmse([0.0, 0.0], [1.0, 1.0]), 1.0)

    def test_deret_kosong(self):
        self.assertEqual(mae([], []), 0.0)
        self.assertEqual(rmse([], []), 0.0)


class TestBuildEvents(unittest.TestCase):
    def test_fitur_hanya_dari_jendela_awal(self):
        """Inti tugasnya: model tidak boleh melihat puncak yang harus ditebak."""
        events = build_events(kejadian("S001", awal_pga=0.3, puncak_pga=0.9))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertAlmostEqual(e["early_peak_pga"], 0.3)
        self.assertAlmostEqual(e["peak_pga"], 0.9)
        # pga_max fitur harus dari jendela awal saja, bukan dari puncak.
        self.assertAlmostEqual(e["features"]["pga_max"], 0.3)

    def test_kejadian_tidak_terpicu_dibuang(self):
        self.assertEqual(build_events(kejadian("S001", awal_pga=0.01, puncak_pga=0.02)), [])

    def test_beberapa_sesi_dan_node(self):
        rows = kejadian("S001", node_id="NODE_A") + kejadian("S002", node_id="NODE_B")
        events = build_events(rows)
        self.assertEqual(len(events), 2)
        self.assertEqual({e["node_id"] for e in events}, {"NODE_A", "NODE_B"})

    def test_baris_tanpa_session_id_diabaikan(self):
        rows = kejadian("S001")
        for r in rows:
            r["session_id"] = None
        self.assertEqual(build_events(rows), [])

    def test_lead_window_menentukan_apa_yang_dilihat(self):
        rows = kejadian("S001", awal_pga=0.3, puncak_pga=0.9)
        luas = build_events(rows, lead_window_s=2.0)[0]
        self.assertAlmostEqual(luas["early_peak_pga"], 0.9,
                               msg="jendela 2 detik sudah memuat puncak")


class TestSplitEvents(unittest.TestCase):
    def _events(self, n=10):
        return [{"session_id": f"S{i:03d}", "peak_pga": 0.5, "early_peak_pga": 0.3}
                for i in range(n)]

    def test_sesi_tidak_muncul_di_dua_sisi(self):
        latih, uji = split_events(self._events())
        self.assertEqual(
            set(e["session_id"] for e in latih) & set(e["session_id"] for e in uji),
            set(),
        )

    def test_uji_tidak_kosong(self):
        _, uji = split_events(self._events())
        self.assertGreaterEqual(len(uji), 1)

    def test_deterministik(self):
        a = split_events(self._events(), seed=5)
        b = split_events(self._events(), seed=5)
        self.assertEqual(a, b)


class TestEvaluateBaselines(unittest.TestCase):
    def test_persistensi_sempurna_bila_tidak_menguat(self):
        events = [{"peak_pga": 0.5, "early_peak_pga": 0.5}]
        hasil = evaluate_baselines(events)["persistensi"]
        self.assertAlmostEqual(hasil["pga_mae"], 0.0)

    def test_persistensi_meleset_bila_menguat(self):
        """Persis kelemahan sistem lama: puncak baru muncul belakangan."""
        events = [{"peak_pga": 0.9, "early_peak_pga": 0.3}]
        hasil = evaluate_baselines(events)["persistensi"]
        self.assertAlmostEqual(hasil["pga_mae"], 0.6)
        self.assertGreater(hasil["magnitudo_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()


class TestMetadataJsonSerializable(unittest.TestCase):
    """Regresi: numpy.bool_ bukan turunan bool, json.dump menolaknya.

    Gejalanya menyesatkan — pelatihan terlihat sukses dan mencetak seluruh
    metrik, tetapi metadata tidak pernah tertulis ke disk.
    """

    def test_bool_numpy_dikonversi(self):
        import json

        import numpy as np

        perbaikan = np.float64(0.05)
        with self.assertRaises(TypeError):
            json.dumps({"beats": perbaikan > 0})
        json.dumps({"beats": bool(perbaikan > 0)})

    def test_float_numpy_bisa_diserialisasi_setelah_dikonversi(self):
        import json

        import numpy as np

        json.dumps({"mae": float(np.float64(0.123))})
