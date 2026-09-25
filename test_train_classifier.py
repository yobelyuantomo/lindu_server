"""Uji ml_training/train_classifier.py.

Bagian yang tidak memerlukan scikit-learn diuji selalu; pelatihan end-to-end
dilewati bila scikit-learn belum terpasang, supaya berkas uji ini tetap bisa
dijalankan di lingkungan yang hanya menjalankan runtime server.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402
from ml_training.train_classifier import (  # noqa: E402
    feature_importance,
    split_xy,
)

try:  # pragma: no cover - bergantung lingkungan
    import sklearn  # noqa: F401
    SKLEARN_ADA = True
except ImportError:  # pragma: no cover
    SKLEARN_ADA = False


def baris(label="noise", augmented=False, **fitur):
    row = {name: 1.0 for name in FEATURE_NAMES}
    row.update(fitur)
    row["label"] = label
    row["is_augmented"] = augmented
    row["split"] = "train"
    return row


class TestSplitXy(unittest.TestCase):
    def test_urutan_fitur_mengikuti_kontrak(self):
        row = baris(**{name: float(i) for i, name in enumerate(FEATURE_NAMES)})
        X, y, _ = split_xy([row])
        self.assertEqual(X[0], [float(i) for i in range(len(FEATURE_NAMES))])
        self.assertEqual(y, ["noise"])

    def test_augmentasi_bisa_dikecualikan(self):
        rows = [baris(augmented=False), baris(augmented=True), baris(augmented=True)]
        _, y_semua, _ = split_xy(rows, include_augmented=True)
        _, y_asli, _ = split_xy(rows, include_augmented=False)
        self.assertEqual(len(y_semua), 3)
        self.assertEqual(len(y_asli), 1)

    def test_fitur_none_jadi_nol(self):
        row = baris()
        row[FEATURE_NAMES[0]] = None
        X, _, _ = split_xy([row])
        self.assertEqual(X[0][0], 0.0)

    def test_dataset_kosong(self):
        X, y, dipakai = split_xy([])
        self.assertEqual((X, y, dipakai), ([], [], []))


class TestFeatureImportance(unittest.TestCase):
    def test_model_tanpa_importance(self):
        class Tanpa:
            pass

        self.assertEqual(feature_importance(Tanpa()), {})

    def test_diurutkan_menurun(self):
        class Dengan:
            feature_importances_ = [0.1 * i for i in range(len(FEATURE_NAMES))]

        hasil = feature_importance(Dengan())
        nilai = list(hasil.values())
        self.assertEqual(nilai, sorted(nilai, reverse=True))
        self.assertEqual(set(hasil), set(FEATURE_NAMES))


@unittest.skipUnless(SKLEARN_ADA, "scikit-learn belum terpasang")
class TestPelatihanEndToEnd(unittest.TestCase):
    def _dataset(self):
        """Dua kelas yang dapat dipisahkan dari freq_mean."""
        rows = []
        for i in range(30):
            rows.append(baris(label="noise", freq_mean=30.0 + i * 0.1, pga_max=0.5))
            rows.append(baris(label="earthquake", freq_mean=6.0 + i * 0.1, pga_max=0.5))
        return rows

    def test_kandidat_bisa_dilatih_dan_memprediksi(self):
        from ml_training.train_classifier import build_candidates

        rows = self._dataset()
        X, y, _ = split_xy(rows)
        for nama, model in build_candidates(seed=1).items():
            model.fit(X, y)
            prediksi = list(model.predict(X))
            self.assertEqual(len(prediksi), len(y), f"{nama} salah jumlah prediksi")
            self.assertTrue(set(prediksi) <= {"noise", "earthquake"})

    def test_memilih_model_dari_split_val(self):
        from ml_training.train_classifier import train_and_select

        rows = self._dataset()
        model, nama, hasil_val = train_and_select(rows, rows, seed=1)
        self.assertIn(nama, hasil_val)
        self.assertIsNotNone(model)
        # Pada data yang terpisah bersih, keduanya harus mendekati sempurna.
        self.assertGreater(hasil_val[nama]["macro_f1"], 0.9)


if __name__ == "__main__":
    unittest.main()
