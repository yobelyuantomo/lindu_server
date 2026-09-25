"""Uji ml_training/metrics.py dan evaluate_baseline.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.evaluate_baseline import (  # noqa: E402
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    evaluate,
    predict_window,
)
from ml_training.metrics import (  # noqa: E402
    accuracy,
    classification_report,
    confusion_matrix,
    false_positive_rate,
    precision_recall_f1,
)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(accuracy(["a", "a", "b"], ["a", "b", "b"]), 2 / 3)
        self.assertEqual(accuracy([], []), 0.0)

    def test_precision_recall_f1_sempurna(self):
        p, r, f = precision_recall_f1(["a", "b"], ["a", "b"], "a")
        self.assertEqual((p, r, f), (1.0, 1.0, 1.0))

    def test_precision_recall_f1_dihitung_manual(self):
        # tp=2, fp=1, fn=1 untuk kelas 'q'
        y_true = ["q", "q", "q", "n"]
        y_pred = ["q", "q", "n", "q"]
        p, r, f = precision_recall_f1(y_true, y_pred, "q")
        self.assertAlmostEqual(p, 2 / 3)
        self.assertAlmostEqual(r, 2 / 3)
        self.assertAlmostEqual(f, 2 / 3)

    def test_tidak_ada_prediksi_positif(self):
        p, r, f = precision_recall_f1(["q", "q"], ["n", "n"], "q")
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_false_positive_rate(self):
        # 3 negatif, 2 di antaranya salah diprediksi positif
        y_true = ["n", "n", "n", "q"]
        y_pred = ["q", "q", "n", "q"]
        self.assertAlmostEqual(false_positive_rate(y_true, y_pred, "q"), 2 / 3)

    def test_fpr_tanpa_negatif(self):
        self.assertEqual(false_positive_rate(["q"], ["q"], "q"), 0.0)

    def test_confusion_matrix(self):
        matrix, labels = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(matrix[("a", "a")], 1)
        self.assertEqual(matrix[("a", "b")], 1)
        self.assertEqual(matrix[("b", "b")], 1)
        self.assertEqual(matrix[("b", "a")], 0)

    def test_classification_report_lengkap(self):
        report = classification_report(["q", "n", "q"], ["q", "n", "n"], "q")
        self.assertEqual(report["n_samples"], 3)
        self.assertIn("q", report["per_class"])
        self.assertIn("n", report["per_class"])
        self.assertEqual(report["per_class"]["q"]["support"], 2)
        self.assertAlmostEqual(report["accuracy"], 2 / 3)


class TestPredictWindow(unittest.TestCase):
    def test_lolos_ketiga_lapis(self):
        fitur = {"pga_max": 0.5, "sta_lta_max": 3.0, "freq_min": 8.0}
        self.assertEqual(predict_window(fitur), POSITIVE_LABEL)

    def test_gagal_karena_pga(self):
        fitur = {"pga_max": 0.05, "sta_lta_max": 3.0, "freq_min": 8.0}
        self.assertEqual(predict_window(fitur), NEGATIVE_LABEL)

    def test_gagal_karena_sta_lta(self):
        fitur = {"pga_max": 0.5, "sta_lta_max": 1.2, "freq_min": 8.0}
        self.assertEqual(predict_window(fitur), NEGATIVE_LABEL)

    def test_gagal_karena_frekuensi_terlalu_tinggi(self):
        """Hentakan kaki: kuat dan berkelanjutan, tapi frekuensinya tinggi."""
        fitur = {"pga_max": 0.9, "sta_lta_max": 4.0, "freq_min": 28.0}
        self.assertEqual(predict_window(fitur), NEGATIVE_LABEL)

    def test_memakai_nilai_ekstrem_bukan_rata_rata(self):
        """Baseline dinilai optimistis: satu sampel lolos sudah cukup.

        Sistem lama memang begitu perilakunya — satu pesan yang lolos filter
        langsung masuk trigger buffer. Memakai rata-rata akan membuat baseline
        terlihat lebih baik daripada yang sebenarnya berjalan.
        """
        fitur = {"pga_max": 0.5, "pga_mean": 0.01, "sta_lta_max": 3.0,
                 "sta_lta_mean": 1.0, "freq_min": 8.0, "freq_mean": 30.0}
        self.assertEqual(predict_window(fitur), POSITIVE_LABEL)

    def test_fitur_hilang_dianggap_negatif(self):
        self.assertEqual(predict_window({}), NEGATIVE_LABEL)


class TestEvaluate(unittest.TestCase):
    def test_baseline_sempurna_pada_data_terpisah_jelas(self):
        rows = [
            {"label": "earthquake", "pga_max": 0.6, "sta_lta_max": 4.0, "freq_min": 6.0},
            {"label": "noise", "pga_max": 0.6, "sta_lta_max": 4.0, "freq_min": 30.0},
        ]
        report, _ = evaluate(rows)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["false_positive_rate"], 0.0)

    def test_baseline_gagal_pada_noise_frekuensi_rendah(self):
        """Persis kasus yang ingin diperbaiki ML: truk berat, getaran
        berfrekuensi rendah yang bukan gempa."""
        rows = [
            {"label": "noise", "pga_max": 0.6, "sta_lta_max": 4.0, "freq_min": 9.0},
            {"label": "noise", "pga_max": 0.7, "sta_lta_max": 3.5, "freq_min": 7.0},
        ]
        report, _ = evaluate(rows)
        self.assertEqual(report["false_positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
