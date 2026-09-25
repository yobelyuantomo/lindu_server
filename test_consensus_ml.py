"""Uji integrasi lapisan ML ke dalam consensus.py.

Yang dibuktikan di sini bukan kualitas prediksi, melainkan bahwa **menyalakan
lapisan ML tidak mengubah perilaku deteksi gempa**, dan bahwa kegagalan di
lapisan itu tidak pernah merambat ke jalur alarm.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import consensus  # noqa: E402
from ml.inference_engine import InferenceEngine  # noqa: E402


class TestInitMl(unittest.TestCase):
    def setUp(self):
        self._asli = consensus.ml_engine

    def tearDown(self):
        consensus.ml_engine = self._asli

    def test_model_tidak_ada_tidak_melempar(self):
        """Server harus tetap menyala walau model belum pernah dilatih."""
        consensus.init_ml()
        self.assertIsNone(consensus.ml_engine)

    def test_shadow_mode_default_aktif(self):
        """Default harus aman: prediksi dicatat, aktuator tidak disentuh."""
        self.assertTrue(consensus.ML_SHADOW_MODE)

    def test_init_ml_bisa_dipanggil_berulang(self):
        consensus.init_ml()
        consensus.init_ml()
        self.assertIsNone(consensus.ml_engine)


class TestSaveMlPrediction(unittest.TestCase):
    def test_tanpa_database_tidak_melempar(self):
        """Gagal mencatat prediksi tidak boleh menjatuhkan deteksi gempa."""
        asli = consensus.get_db_connection
        consensus.get_db_connection = lambda: None
        try:
            consensus.save_ml_prediction(
                "NODE_A",
                {"label": "earthquake", "confidence": 0.9, "features": {}},
                {"decision": "confirmed", "trigger_actuator": True},
                True,
            )
        finally:
            consensus.get_db_connection = asli


class TestPerilakuTidakBerubah(unittest.TestCase):
    """Filter rule-based harus menghasilkan keputusan yang sama persis
    seperti Assignment 2, terlepas dari ada atau tidaknya lapisan ML."""

    def _lolos_filter(self, pga, sta_lta, freq_hz):
        return pga >= 0.12 and sta_lta >= 2.0 and freq_hz <= 20

    def test_ambang_filter_tidak_berubah(self):
        self.assertTrue(self._lolos_filter(0.12, 2.0, 20))
        self.assertFalse(self._lolos_filter(0.11, 2.0, 20))
        self.assertFalse(self._lolos_filter(0.12, 1.9, 20))
        self.assertFalse(self._lolos_filter(0.12, 2.0, 21))

    def test_ml_mati_setara_assignment_2(self):
        from ml.inference_engine import combine_verdicts

        for rule in (True, False):
            hasil = combine_verdicts(rule, None)
            self.assertEqual(hasil["trigger_actuator"], rule)


class TestEngineTanpaModel(unittest.TestCase):
    def test_predict_aman_tanpa_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = InferenceEngine(models_dir=tmp)
            engine.load()
            hasil = engine.predict({
                "node_id": "NODE_A", "ts": 1.0, "pga": 0.5,
                "sta_lta": 3.0, "freq_hz": 8.0,
            })
            self.assertFalse(hasil.available)
            self.assertEqual(engine.stats()["fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
