"""Uji ml/drift_monitor.py (Population Stability Index)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.drift_monitor import (  # noqa: E402
    MIN_SAMPLES,
    PSI_SIGNIFIKAN,
    build_baseline,
    check_drift,
    klasifikasi,
    psi,
)


def nilai(mean, spread=1.0, n=500, seed=1):
    rng = random.Random(seed)
    return [rng.gauss(mean, spread) for _ in range(n)]


class TestBuildBaseline(unittest.TestCase):
    def test_proporsi_berjumlah_satu(self):
        b = build_baseline(nilai(0.0))
        self.assertAlmostEqual(sum(b["proportions"]), 1.0, places=6)

    def test_fitur_konstan(self):
        b = build_baseline([5.0] * 100)
        self.assertEqual(b["constant"], 5.0)
        self.assertIsNone(b["proportions"])

    def test_terlalu_sedikit_nilai(self):
        self.assertIsNone(build_baseline([1.0]))
        self.assertIsNone(build_baseline([]))

    def test_none_diabaikan(self):
        b = build_baseline([1.0, None, 2.0, None, 3.0])
        self.assertIsNotNone(b)


class TestPsi(unittest.TestCase):
    def test_distribusi_sama_psi_mendekati_nol(self):
        data = nilai(0.0, seed=1)
        self.assertLess(psi(build_baseline(data), data), 0.01)

    def test_pergeseran_besar_terdeteksi(self):
        baseline = build_baseline(nilai(0.0, seed=1))
        bergeser = nilai(5.0, seed=2)
        self.assertGreater(psi(baseline, bergeser), PSI_SIGNIFIKAN)

    def test_pergeseran_kecil_nilainya_kecil(self):
        baseline = build_baseline(nilai(0.0, spread=1.0, seed=1))
        sedikit = nilai(0.05, spread=1.0, seed=2)
        self.assertLess(psi(baseline, sedikit), PSI_SIGNIFIKAN)

    def test_sampel_terlalu_sedikit_mengembalikan_none(self):
        """'Tidak tahu' harus dibedakan dari 'stabil'."""
        baseline = build_baseline(nilai(0.0))
        self.assertIsNone(psi(baseline, nilai(0.0, n=MIN_SAMPLES - 1)))

    def test_baseline_kosong_atau_konstan(self):
        self.assertIsNone(psi(None, nilai(0.0)))
        self.assertIsNone(psi(build_baseline([5.0] * 100), nilai(0.0)))

    def test_bin_kosong_tidak_menghasilkan_infinity(self):
        """Bin kosong menghasilkan log(0) kalau tidak diberi lantai."""
        baseline = build_baseline(nilai(0.0, spread=1.0, seed=1))
        jauh = [100.0] * 200
        hasil = psi(baseline, jauh)
        self.assertIsNotNone(hasil)
        self.assertTrue(hasil == hasil, "PSI tidak boleh NaN")
        self.assertNotEqual(hasil, float("inf"))


class TestKlasifikasi(unittest.TestCase):
    def test_batas(self):
        self.assertEqual(klasifikasi(None), "tidak diketahui")
        self.assertEqual(klasifikasi(0.05), "stabil")
        self.assertEqual(klasifikasi(0.15), "bergeser")
        self.assertEqual(klasifikasi(0.50), "signifikan")


class TestCheckDrift(unittest.TestCase):
    def setUp(self):
        self.baselines = {
            "pga_max": build_baseline(nilai(0.5, seed=1)),
            "freq_mean": build_baseline(nilai(10.0, seed=2)),
        }

    def test_semua_stabil(self):
        hasil = check_drift(self.baselines, {
            "pga_max": nilai(0.5, seed=11),
            "freq_mean": nilai(10.0, seed=12),
        })
        self.assertEqual(hasil["status"], "stabil")
        self.assertEqual(hasil["drifted_features"], [])
        self.assertIn("Tidak ada tindakan", hasil["recommendation"])

    def test_satu_fitur_bergeser(self):
        hasil = check_drift(self.baselines, {
            "pga_max": nilai(0.5, seed=11),
            "freq_mean": nilai(40.0, seed=12),
        })
        self.assertEqual(hasil["status"], "signifikan")
        self.assertEqual(hasil["drifted_features"], ["freq_mean"])
        self.assertIn("Latih ulang", hasil["recommendation"])

    def test_sampel_kurang_dilaporkan_tidak_diketahui(self):
        hasil = check_drift(self.baselines, {"pga_max": [0.5] * 5})
        self.assertEqual(hasil["status"], "tidak diketahui")
        self.assertEqual(hasil["n_measured"], 0)
        self.assertEqual(hasil["n_unknown"], 2)
        self.assertIn("Belum cukup sampel", hasil["recommendation"])

    def test_fitur_tanpa_sampel_tidak_membuat_crash(self):
        hasil = check_drift(self.baselines, {})
        self.assertEqual(hasil["status"], "tidak diketahui")


if __name__ == "__main__":
    unittest.main()
