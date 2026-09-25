"""Uji ml_training/generate_report.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.generate_report import _delta, susun  # noqa: E402

MODEL_META = {
    "model_type": "random_forest",
    "model_version": "20260925-000000",
    "selection_metric": "macro_f1 pada split val",
    "seed": 42,
    "n_features": 21,
    "n_train_rows": 144, "n_val_rows": 12, "n_test_rows": 12,
    "feature_importance": {"freq_mean": 0.31, "hv_ratio": 0.22},
    "test_report": {
        "accuracy": 1.0, "macro_f1": 1.0, "false_positive_rate": 0.0,
        "n_samples": 12, "labels": ["earthquake", "noise"],
        "per_class": {
            "earthquake": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 6},
            "noise": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 6},
        },
        "confusion_matrix": {
            "earthquake->earthquake": 6, "earthquake->noise": 0,
            "noise->earthquake": 0, "noise->noise": 6,
        },
    },
}

BASELINE = {
    "accuracy": 0.5, "macro_f1": 0.3333, "false_positive_rate": 1.0,
    "n_samples": 12, "labels": ["earthquake", "noise"],
    "per_class": {
        "earthquake": {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 6},
        "noise": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 6},
    },
    "confusion_matrix": {
        "earthquake->earthquake": 6, "earthquake->noise": 0,
        "noise->earthquake": 6, "noise->noise": 0,
    },
}

MAG_META = {
    "lead_window_s": 1.0, "n_train_events": 17, "n_test_events": 7,
    "beats_persistence": True,
    "model_report": {"pga_mae": 0.0084, "pga_rmse": 0.0113, "magnitudo_mae": 0.0067},
    "baselines": {"persistensi": {
        "pga_mae": 0.0473, "pga_rmse": 0.0622,
        "magnitudo_mae": 0.0399, "magnitudo_rmse": 0.0537,
    }},
}


class TestDelta(unittest.TestCase):
    def test_lebih_besar_lebih_baik(self):
        self.assertIn("✅", _delta(1.0, 0.5, True))
        self.assertIn("❌", _delta(0.5, 1.0, True))

    def test_lebih_kecil_lebih_baik(self):
        """FP rate turun adalah kemenangan, walau selisihnya negatif."""
        self.assertIn("✅", _delta(0.0, 1.0, False))
        self.assertIn("❌", _delta(1.0, 0.0, False))

    def test_tanpa_selisih_tidak_diberi_penanda(self):
        hasil = _delta(0.5, 0.5, True)
        self.assertNotIn("✅", hasil)
        self.assertNotIn("❌", hasil)

    def test_nilai_hilang(self):
        self.assertEqual(_delta(None, 0.5, True), "—")
        self.assertEqual(_delta(0.5, None, True), "—")


class TestSusun(unittest.TestCase):
    def setUp(self):
        self.isi = susun(MODEL_META, BASELINE, MAG_META)

    def test_seluruh_bagian_hadir(self):
        for judul in ("Ringkasan Perbandingan", "Rincian per Kelas",
                      "Confusion matrix", "Model",
                      "Estimasi Dini Puncak Guncangan", "Keterbatasan"):
            self.assertIn(judul, self.isi)

    def test_angka_kedua_sistem_muncul(self):
        self.assertIn("0.5000", self.isi)   # accuracy rule-based
        self.assertIn("1.0000", self.isi)   # accuracy ML

    def test_keterbatasan_menyebut_data_peragaan(self):
        """Hal yang paling mudah tergoda untuk disembunyikan."""
        self.assertIn("peragaan", self.isi)
        self.assertIn("bukan gempa tektonik", self.isi)

    def test_keterbatasan_menyebut_jendela_terpicu(self):
        self.assertIn("terpicu", self.isi)

    def test_catatan_peringatan_muncul_di_atas(self):
        isi = susun(MODEL_META, BASELINE, MAG_META, catatan="DATA SINTETIS")
        posisi = isi.index("DATA SINTETIS")
        self.assertLess(posisi, isi.index("Ringkasan Perbandingan"))

    def test_tanpa_metadata_magnitudo_tetap_tersusun(self):
        isi = susun(MODEL_META, BASELINE, None)
        self.assertNotIn("Estimasi Dini Puncak", isi)
        self.assertIn("Ringkasan Perbandingan", isi)

    def test_menyuruh_tidak_menyunting_tangan(self):
        self.assertIn("Jangan menyunting berkas ini dengan tangan", self.isi)

    def test_laporan_kosong_tidak_membuat_crash(self):
        isi = susun({}, {}, None)
        self.assertIn("Ringkasan Perbandingan", isi)
        self.assertIn("—", isi)


if __name__ == "__main__":
    unittest.main()
