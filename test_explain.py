"""Uji ml/explain.py (SHAP + penilaian kesehatan sensor)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.explain import (  # noqa: E402
    SENSOR_FAULT_SECONDS,
    Explainer,
    SensorHealthTracker,
    _ambil_baris,
    ringkas_alasan,
)
from ml.feature_extractor import FEATURE_NAMES  # noqa: E402

try:  # pragma: no cover
    import shap  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier
    SHAP_ADA = True
except ImportError:  # pragma: no cover
    SHAP_ADA = False


class TestExplainerTanpaShap(unittest.TestCase):
    def test_model_tidak_valid_tidak_melempar(self):
        """Penjelasan bersifat tambahan; kegagalannya tidak boleh merambat."""
        e = Explainer(object())
        self.assertFalse(e.ready)
        self.assertEqual(e.explain([0.0] * len(FEATURE_NAMES)), [])
        self.assertIsNotNone(e.error)


class TestAmbilBaris(unittest.TestCase):
    """Bentuk keluaran SHAP berbeda antar versi; ketiganya harus ditangani."""

    def test_bentuk_list_per_kelas(self):
        nilai = [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]
        self.assertEqual(_ambil_baris(nilai, 1), [4.0, 5.0, 6.0])

    def test_class_index_di_luar_jangkauan_pakai_yang_pertama(self):
        nilai = [[[1.0, 2.0]]]
        self.assertEqual(_ambil_baris(nilai, 9), [1.0, 2.0])

    def test_bentuk_tidak_dikenal(self):
        self.assertIsNone(_ambil_baris(None, 0))
        self.assertIsNone(_ambil_baris(42, 0))


class TestRingkasAlasan(unittest.TestCase):
    def test_kosong(self):
        self.assertEqual(ringkas_alasan([]), "penjelasan tidak tersedia")

    def test_menyebut_tiga_teratas(self):
        kontribusi = [
            {"feature": "freq_mean", "contribution": 0.5},
            {"feature": "hv_ratio", "contribution": -0.3},
            {"feature": "pga_max", "contribution": 0.2},
            {"feature": "energy_cumulative", "contribution": 0.1},
        ]
        teks = ringkas_alasan(kontribusi)
        self.assertIn("freq_mean", teks)
        self.assertIn("hv_ratio", teks)
        self.assertNotIn("energy_cumulative", teks)


class TestSensorHealthTracker(unittest.TestCase):
    def setUp(self):
        self.t = SensorHealthTracker(threshold=0.6, fault_seconds=300.0)

    def test_skor_rendah_normal(self):
        h = self.t.update("NODE_A", 0.1, now=0.0)
        self.assertEqual(h["status"], "normal")
        self.assertFalse(h["is_sensor_fault"])

    def test_anomali_singkat_bukan_sensor_rusak(self):
        """Gempa berlangsung puluhan detik — tidak boleh disalahartikan."""
        self.t.update("NODE_A", 0.9, now=0.0)
        h = self.t.update("NODE_A", 0.9, now=45.0)
        self.assertEqual(h["status"], "anomali")
        self.assertFalse(h["is_sensor_fault"])

    def test_anomali_bertahan_lama_dicurigai_sensor(self):
        self.t.update("NODE_A", 0.9, now=0.0)
        h = self.t.update("NODE_A", 0.9, now=SENSOR_FAULT_SECONDS + 1)
        self.assertTrue(h["is_sensor_fault"])
        self.assertIn("pemasangan fisik", h["recommendation"])

    def test_kembali_normal_mereset_hitungan(self):
        self.t.update("NODE_A", 0.9, now=0.0)
        self.t.update("NODE_A", 0.1, now=100.0)     # tenang -> reset
        h = self.t.update("NODE_A", 0.9, now=200.0)  # anomali lagi
        self.assertEqual(h["status"], "anomali")
        self.assertAlmostEqual(h["anomalous_for_s"], 0.0)

    def test_node_dilacak_terpisah(self):
        self.t.update("NODE_A", 0.9, now=0.0)
        h = self.t.update("NODE_B", 0.9, now=SENSOR_FAULT_SECONDS + 1)
        self.assertFalse(h["is_sensor_fault"], "NODE_B baru mulai anomali")

    def test_skor_tidak_ada(self):
        h = self.t.update("NODE_A", None, now=0.0)
        self.assertEqual(h["status"], "tidak diketahui")

    def test_reset(self):
        self.t.update("NODE_A", 0.9, now=0.0)
        self.t.reset("NODE_A")
        h = self.t.update("NODE_A", 0.9, now=SENSOR_FAULT_SECONDS + 1)
        self.assertFalse(h["is_sensor_fault"])


@unittest.skipUnless(SHAP_ADA, "shap/scikit-learn belum terpasang")
class TestExplainerDenganShap(unittest.TestCase):
    def setUp(self):
        rng = random.Random(5)
        X, y = [], []
        for _ in range(60):
            X.append([rng.gauss(0, 1) for _ in FEATURE_NAMES])
            y.append("noise")
            X.append([rng.gauss(4, 1) for _ in FEATURE_NAMES])
            y.append("earthquake")
        self.model = RandomForestClassifier(n_estimators=10, random_state=1)
        self.model.fit(X, y)
        self.explainer = Explainer(self.model)

    def test_siap(self):
        self.assertTrue(self.explainer.ready)
        self.assertIsNone(self.explainer.error)

    def test_menghasilkan_kontribusi(self):
        kontribusi = self.explainer.explain([4.0] * len(FEATURE_NAMES), class_index=0)
        self.assertGreater(len(kontribusi), 0)
        self.assertLessEqual(len(kontribusi), 5)
        for k in kontribusi:
            self.assertIn(k["feature"], FEATURE_NAMES)
            self.assertIn(k["direction"], ("mendukung", "menyanggah"))

    def test_terurut_menurut_besar_kontribusi(self):
        kontribusi = self.explainer.explain([4.0] * len(FEATURE_NAMES))
        besar = [abs(k["contribution"]) for k in kontribusi]
        self.assertEqual(besar, sorted(besar, reverse=True))

    def test_vektor_salah_panjang_tidak_melempar(self):
        self.assertEqual(self.explainer.explain([0.0, 1.0]), [])


if __name__ == "__main__":
    unittest.main()
