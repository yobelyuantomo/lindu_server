"""Uji ml_training/label_dataset.py (pelabelan berbasis rentang waktu)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.label_dataset import (  # noqa: E402
    build_session_index,
    label_for,
    label_rows,
    parse_row,
    summarize_windows,
)


def sesi(node_id="NODE_A", label="noise", start=100.0, end=200.0, scenario="melompat"):
    return {
        "node_id": node_id, "label": label, "scenario": scenario,
        "start_ts": str(start), "end_ts": str(end), "notes": "",
    }


def row(node_id="NODE_A", ts=150.0, pga=0.5, label=None):
    r = {
        "node_id": node_id, "sensor_ts": ts, "pga": pga, "sta_lta": 3.0,
        "freq_hz": 8.0, "accel_x": 0.1, "accel_y": 0.1, "accel_z": 0.1,
        "temperature": 28.0, "pressure": 1010.0, "humidity": 60.0, "gas_raw": 100.0,
    }
    if label:
        r["label"] = label
    return r


class TestParseRow(unittest.TestCase):
    def test_konversi_numerik(self):
        hasil = parse_row({"node_id": "A", "pga": "0.5", "freq_hz": "12"})
        self.assertEqual(hasil["pga"], 0.5)
        self.assertEqual(hasil["freq_hz"], 12.0)
        self.assertEqual(hasil["node_id"], "A")

    def test_kosong_jadi_none_bukan_nol(self):
        """Membedakan 'tidak terkirim' dari 'bernilai nol'.

        freq_hz 0 lolos filter rule-based, None tidak — jadi keliru di sini
        akan diam-diam mengubah label baseline.
        """
        hasil = parse_row({"freq_hz": "", "pga": None})
        self.assertIsNone(hasil["freq_hz"])
        self.assertIsNone(hasil["pga"])

    def test_nilai_tidak_valid_jadi_none(self):
        self.assertIsNone(parse_row({"pga": "entah"})["pga"])


class TestLabelFor(unittest.TestCase):
    def setUp(self):
        self.index = build_session_index([
            sesi(start=100.0, end=200.0, label="noise"),
            sesi(start=300.0, end=400.0, label="earthquake"),
        ])

    def test_di_dalam_rentang(self):
        self.assertEqual(label_for(self.index, "NODE_A", 150.0), "noise")
        self.assertEqual(label_for(self.index, "NODE_A", 350.0), "earthquake")

    def test_batas_bawah_inklusif_batas_atas_eksklusif(self):
        self.assertEqual(label_for(self.index, "NODE_A", 100.0), "noise")
        self.assertIsNone(label_for(self.index, "NODE_A", 200.0))

    def test_di_luar_rentang(self):
        self.assertIsNone(label_for(self.index, "NODE_A", 250.0))

    def test_node_lain_tidak_kebagian_label(self):
        self.assertIsNone(label_for(self.index, "NODE_B", 150.0))

    def test_sensor_ts_hilang(self):
        self.assertIsNone(label_for(self.index, "NODE_A", None))


class TestLabelRows(unittest.TestCase):
    def test_baris_di_luar_sesi_dibuang(self):
        index = build_session_index([sesi(start=100.0, end=200.0)])
        rows = [row(ts=150.0), row(ts=250.0), row(ts=120.0)]
        labeled, dibuang = label_rows(rows, index)
        self.assertEqual(len(labeled), 2)
        self.assertEqual(dibuang, 1)
        self.assertTrue(all(r["label"] == "noise" for r in labeled))

    def test_tidak_mengubah_baris_asli(self):
        index = build_session_index([sesi(start=100.0, end=200.0)])
        asli = row(ts=150.0)
        label_rows([asli], index)
        self.assertNotIn("label", asli)


class TestSummarizeWindows(unittest.TestCase):
    def test_jendela_terpicu_dihitung_layak(self):
        rows = [row(ts=100.0 + i * 0.1, pga=0.5, label="noise") for i in range(20)]
        layak, tidak = summarize_windows(rows, window_seconds=2.0)
        self.assertEqual(layak["noise"], 1)
        self.assertEqual(sum(tidak.values()), 0)

    def test_jendela_tenang_tidak_dihitung_layak(self):
        rows = [row(ts=100.0 + i * 0.1, pga=0.01, label="noise") for i in range(20)]
        layak, tidak = summarize_windows(rows, window_seconds=2.0)
        self.assertEqual(sum(layak.values()), 0)
        self.assertEqual(tidak["noise"], 1)

    def test_jendela_campur_label_dibuang(self):
        """Jendela yang menyeberangi dua sesi berbeda labelnya ambigu."""
        rows = [row(ts=100.0 + i * 0.1, pga=0.5, label="noise") for i in range(10)]
        rows += [row(ts=101.0 + i * 0.1, pga=0.5, label="earthquake") for i in range(10)]
        layak, tidak = summarize_windows(rows, window_seconds=2.0)
        self.assertEqual(sum(layak.values()) + sum(tidak.values()), 0)

    def test_node_dipisah(self):
        rows = [row(node_id="NODE_A", ts=100.0 + i * 0.1, pga=0.5, label="noise") for i in range(20)]
        rows += [row(node_id="NODE_B", ts=100.0 + i * 0.1, pga=0.5, label="earthquake") for i in range(20)]
        layak, _ = summarize_windows(rows, window_seconds=2.0)
        self.assertEqual(layak["noise"], 1)
        self.assertEqual(layak["earthquake"], 1)


if __name__ == "__main__":
    unittest.main()
