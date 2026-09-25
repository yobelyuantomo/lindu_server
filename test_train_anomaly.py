"""Uji ml_training/train_anomaly.py."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402
from ml_training.train_anomaly import (  # noqa: E402
    MIN_NORMAL_PER_NODE,
    kelompokkan_per_node,
    latih_per_node,
    roc_auc,
)


def baris(label="noise", node_id="NODE_A", geser=0.0, seed=None):
    rng = random.Random(seed)
    row = {n: rng.gauss(1.0 + geser, 0.1) for n in FEATURE_NAMES}
    row["label"] = label
    row["node_id"] = node_id
    row["is_augmented"] = False
    return row


class TestRocAuc(unittest.TestCase):
    def test_pemisahan_sempurna(self):
        self.assertEqual(roc_auc([0.1, 0.2, 0.3], [0.8, 0.9, 1.0]), 1.0)

    def test_terbalik_sepenuhnya(self):
        self.assertEqual(roc_auc([0.8, 0.9, 1.0], [0.1, 0.2, 0.3]), 0.0)

    def test_tidak_terpisah(self):
        self.assertAlmostEqual(roc_auc([1.0, 1.0], [1.0, 1.0]), 0.5)

    def test_seri_ditangani_dengan_peringkat_rata(self):
        """Tanpa peringkat rata-rata, nilai seri membuat AUC bias."""
        self.assertAlmostEqual(roc_auc([0.5, 0.5], [0.5, 0.9]), 0.75)

    def test_deret_kosong(self):
        self.assertIsNone(roc_auc([], [0.5]))
        self.assertIsNone(roc_auc([0.5], []))


class TestKelompokkanPerNode(unittest.TestCase):
    def test_dipisah_per_node_dan_label(self):
        rows = [
            baris(label="noise", node_id="NODE_A"),
            baris(label="noise", node_id="NODE_A"),
            baris(label="earthquake", node_id="NODE_A"),
            baris(label="noise", node_id="NODE_B"),
        ]
        hasil = kelompokkan_per_node(rows)
        self.assertEqual(len(hasil["NODE_A"]["normal"]), 2)
        self.assertEqual(len(hasil["NODE_A"]["novel"]), 1)
        self.assertEqual(len(hasil["NODE_B"]["normal"]), 1)

    def test_baris_tanpa_node_id_diabaikan(self):
        rows = [baris()]
        rows[0]["node_id"] = None
        self.assertEqual(len(kelompokkan_per_node(rows)), 0)

    def test_label_lain_diabaikan(self):
        rows = [baris(label="gas_leak", node_id="NODE_A")]
        hasil = kelompokkan_per_node(rows)
        self.assertEqual(len(hasil["NODE_A"]["normal"]), 0)
        self.assertEqual(len(hasil["NODE_A"]["novel"]), 0)


class TestLatihPerNode(unittest.TestCase):
    def test_node_dengan_data_kurang_dilewati(self):
        per_node = {"NODE_A": {
            "normal": [baris(seed=i) for i in range(MIN_NORMAL_PER_NODE - 1)],
            "novel": [],
        }}
        model, laporan = latih_per_node(per_node)
        self.assertEqual(model, {})
        self.assertEqual(laporan["NODE_A"]["status"], "dilewati")
        self.assertIn("minimal", laporan["NODE_A"]["alasan"])

    def test_gempa_dinilai_lebih_anomali_daripada_noise(self):
        """Inti pendekatannya: model yang tak pernah melihat gempa harus
        menilainya sebagai sesuatu yang di luar kebiasaan."""
        per_node = {"NODE_A": {
            "normal": [baris(geser=0.0, seed=i) for i in range(60)],
            "novel": [baris(geser=5.0, seed=100 + i) for i in range(20)],
        }}
        _, laporan = latih_per_node(per_node)
        r = laporan["NODE_A"]
        self.assertEqual(r["status"], "dilatih")
        self.assertGreater(r["roc_auc"], 0.9)
        self.assertGreater(r["skor_novel_rata"], r["skor_normal_rata"])

    def test_baseline_dibentuk_terpisah_per_node(self):
        """Node ramai tidak boleh dibandingkan dengan node tenang."""
        per_node = {
            "NODE_RAMAI": {"normal": [baris(geser=5.0, seed=i) for i in range(60)], "novel": []},
            "NODE_TENANG": {"normal": [baris(geser=0.0, seed=i) for i in range(60)], "novel": []},
        }
        model, laporan = latih_per_node(per_node)
        self.assertEqual(set(model), {"NODE_RAMAI", "NODE_TENANG"})
        for r in laporan.values():
            self.assertEqual(r["status"], "dilatih")

    def test_tanpa_data_novel_auc_none(self):
        per_node = {"NODE_A": {
            "normal": [baris(seed=i) for i in range(60)], "novel": [],
        }}
        _, laporan = latih_per_node(per_node)
        self.assertIsNone(laporan["NODE_A"]["roc_auc"])
        self.assertEqual(laporan["NODE_A"]["status"], "dilatih")


if __name__ == "__main__":
    unittest.main()
