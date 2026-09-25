"""Uji ml_training/export_edge_model.py (generator kode C untuk ESP32)."""

import os
import random
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402
from ml_training.export_edge_model import (  # noqa: E402
    LEAF,
    _c_float,
    extract_trees,
    generate_header,
    predict_with_trees,
)

try:  # pragma: no cover - bergantung lingkungan
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_ADA = True
except ImportError:  # pragma: no cover
    SKLEARN_ADA = False


class TestCFloat(unittest.TestCase):
    """Regresi bug yang ditemukan compiler xtensa.

    ``f"{-2.0:.6g}"`` menghasilkan ``"-2"``, dan ``-2f`` adalah konstanta
    bilangan bulat dengan akhiran ``f`` — ditolak compiler C. Nilai ambang
    daun di sklearn tepat -2.0, jadi kasus ini muncul di SETIAP model.
    """

    LITERAL_C = re.compile(r"^-?(\d+\.\d*|\d*\.\d+|\d+\.?\d*[eE][+-]?\d+)f$")

    def test_bilangan_bulat_tetap_jadi_literal_float(self):
        for nilai in (-2.0, 0.0, 1.0, 42.0, -100.0):
            hasil = _c_float(nilai)
            self.assertRegex(hasil, self.LITERAL_C, f"{nilai} -> {hasil}")

    def test_pecahan_tetap_sah(self):
        for nilai in (1.08718, -0.5, 27.5884):
            self.assertRegex(_c_float(nilai), self.LITERAL_C)

    def test_notasi_eksponen_tidak_dirusak(self):
        hasil = _c_float(1e-8)
        self.assertRegex(hasil, self.LITERAL_C)
        self.assertNotIn("..", hasil)

    def test_nilai_daun_sklearn(self):
        """Nilai persis yang membuat kompilasi gagal sebelum diperbaiki."""
        self.assertEqual(_c_float(-2.0), "-2.0f")


@unittest.skipUnless(SKLEARN_ADA, "scikit-learn belum terpasang")
class TestEkspor(unittest.TestCase):
    def setUp(self):
        rng = random.Random(3)
        X, y = [], []
        for _ in range(80):
            X.append([rng.gauss(0, 1) for _ in FEATURE_NAMES])
            y.append("noise")
            X.append([rng.gauss(4, 1) for _ in FEATURE_NAMES])
            y.append("earthquake")
        self.model = RandomForestClassifier(n_estimators=8, random_state=1)
        self.model.fit(X, y)
        self.X = X
        self.classes = [str(c) for c in self.model.classes_]
        self.pohon = extract_trees(self.model)

    def test_struktur_pohon(self):
        self.assertEqual(len(self.pohon), 8)
        for t in self.pohon:
            n = len(t["feature"])
            self.assertEqual(len(t["threshold"]), n)
            self.assertEqual(len(t["left"]), n)
            self.assertEqual(len(t["right"]), n)
            self.assertEqual(len(t["value"]), n)

    def test_daun_ditandai(self):
        for t in self.pohon:
            for i, f in enumerate(t["feature"]):
                punya_anak = t["left"][i] != -1
                self.assertEqual(f != LEAF, punya_anak)

    def test_nilai_daun_ternormalisasi(self):
        for t in self.pohon:
            for baris in t["value"]:
                self.assertAlmostEqual(sum(baris), 1.0, places=6)

    def test_kesetaraan_dengan_sklearn(self):
        """Inti pengujiannya: array yang diekspor harus berperilaku identik."""
        diharapkan = [str(v) for v in self.model.predict(self.X)]
        didapat = [predict_with_trees(self.pohon, self.classes, v)[0] for v in self.X]
        self.assertEqual(didapat, diharapkan)

    def test_kesetaraan_pada_vektor_acak(self):
        rng = random.Random(99)
        vektor = [[rng.uniform(-8, 8) for _ in FEATURE_NAMES] for _ in range(300)]
        diharapkan = [str(v) for v in self.model.predict(vektor)]
        didapat = [predict_with_trees(self.pohon, self.classes, v)[0] for v in vektor]
        self.assertEqual(didapat, diharapkan)

    def test_confidence_dalam_rentang_sah(self):
        for v in self.X[:20]:
            _, conf = predict_with_trees(self.pohon, self.classes, v)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_header_memuat_kontrak(self):
        isi = generate_header(self.pohon, self.classes, "uji-1", FEATURE_NAMES)
        for penanda in ("#pragma once", "EDGE_MODEL_VERSION", "EDGE_N_TREES",
                        "EDGE_N_CLASSES", "edge_model_predict",
                        "JANGAN DISUNTING DENGAN TANGAN"):
            self.assertIn(penanda, isi)

    def test_header_memuat_urutan_fitur(self):
        """Firmware harus mengisi vektor dengan urutan yang sama persis."""
        isi = generate_header(self.pohon, self.classes, "uji-1", FEATURE_NAMES)
        posisi = [isi.index(f'"{n}"') for n in FEATURE_NAMES]
        self.assertEqual(posisi, sorted(posisi), "urutan fitur tidak terjaga")

    def test_seluruh_literal_float_sah(self):
        """Menangkap ulang bug -2f tanpa perlu menjalankan compiler."""
        isi = generate_header(self.pohon, self.classes, "uji-1", FEATURE_NAMES)
        for blok in ("EDGE_THRESHOLD", "EDGE_VALUE"):
            potong = isi.split(f"{blok}[] = {{")[1].split("};")[0]
            for token in re.findall(r"-?[\w.+-]+f(?=[,\s])", potong):
                self.assertRegex(token, TestCFloat.LITERAL_C, f"{blok}: {token}")


if __name__ == "__main__":
    unittest.main()
