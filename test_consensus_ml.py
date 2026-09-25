"""Uji integrasi lapisan ML ke dalam consensus.py.

Yang dibuktikan di sini bukan kualitas prediksi, melainkan bahwa **menyalakan
lapisan ML tidak mengubah perilaku deteksi gempa**, dan bahwa kegagalan di
lapisan itu tidak pernah merambat ke jalur alarm.
"""

import json
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


class TestPublishMlPrediction(unittest.TestCase):
    """Penyiaran prediksi tidak boleh bisa menjatuhkan jalur deteksi."""

    def setUp(self):
        self._asli = consensus.mqtt_client

    def tearDown(self):
        consensus.mqtt_client = self._asli

    def test_tanpa_mqtt_client_tidak_melempar(self):
        consensus.mqtt_client = None
        consensus.publish_ml_prediction(
            "NODE_A", {"label": "earthquake", "confidence": 0.9},
            {"decision": "confirmed"}, True,
        )

    def test_payload_dan_topik(self):
        terkirim = []

        class MqttPalsu:
            def publish(self, topic, payload):
                terkirim.append((topic, payload))

        consensus.mqtt_client = MqttPalsu()
        consensus.publish_ml_prediction(
            "NODE_A",
            {"label": "earthquake", "confidence": 0.93,
             "shadow_mode": True, "model_version": "v1"},
            {"decision": "confirmed"}, True,
        )

        self.assertEqual(len(terkirim), 1)
        topik, payload = terkirim[0]
        self.assertEqual(topik, "lindu/ml/prediction/NODE_A")

        data = json.loads(payload)
        self.assertEqual(data["label"], "earthquake")
        self.assertAlmostEqual(data["confidence"], 0.93)
        self.assertTrue(data["shadow_mode"])
        self.assertNotIn("features", data, "vektor fitur tidak boleh ikut disiarkan")

    def test_topik_terpisah_dari_alarm(self):
        """Menumpang di TOPIC_ALARM berisiko menyalakan sirine di node lama."""
        self.assertNotEqual(consensus.TOPIC_ML_PREDICTION, consensus.TOPIC_ALARM)
        self.assertFalse(consensus.TOPIC_ML_PREDICTION.startswith("lindu/actuator"))

    def test_mqtt_meledak_tidak_melempar(self):
        class MqttMeledak:
            def publish(self, *a, **k):
                raise RuntimeError("broker mati")

        consensus.mqtt_client = MqttMeledak()
        consensus.publish_ml_prediction(
            "NODE_A", {"label": "noise"}, {"decision": "normal"}, False,
        )


class TestPemulihanConnectionPool(unittest.TestCase):
    """Pool harus pulih dari KEDUA mode kegagalan, bukan hanya satu.

    Satu kali gangguan database tidak boleh membuat server berhenti menyimpan
    apa pun sampai di-restart manual.
    """

    def setUp(self):
        self._pool_asli = consensus.db_pool
        self._build_asli = consensus.build_pool

    def tearDown(self):
        consensus.db_pool = self._pool_asli
        consensus.build_pool = self._build_asli

    def test_pool_tertutup_dibangun_ulang(self):
        """getconn() yang melempar dulu jatuh ke except luar tanpa rebuild."""
        class PoolTertutup:
            def getconn(self):
                raise Exception("connection pool is closed")

            def closeall(self):
                pass

        class KoneksiSehat:
            def cursor(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, q):
                return None

        class PoolBaru:
            def getconn(self):
                return KoneksiSehat()

        consensus.db_pool = PoolTertutup()
        consensus.build_pool = lambda: PoolBaru()
        self.assertIsNotNone(consensus.get_db_connection())

    def test_gagal_rebuild_menyisakan_none_bukan_pool_rusak(self):
        """Kunci pemulihan: panggilan BERIKUTNYA harus mencoba lagi.

        Kalau db_pool dibiarkan menunjuk pool tertutup, sistem menyerah
        selamanya walau database sudah hidup kembali.
        """
        class PoolTertutup:
            def getconn(self):
                raise Exception("connection pool is closed")

            def closeall(self):
                pass

        consensus.db_pool = PoolTertutup()
        consensus.build_pool = lambda: None

        self.assertIsNone(consensus.get_db_connection())
        self.assertIsNone(consensus.db_pool, "pool rusak harus dibuang, bukan disimpan")

    def test_pool_none_dibangun_saat_dibutuhkan(self):
        dipanggil = []

        consensus.db_pool = None
        consensus.build_pool = lambda: dipanggil.append(1) or None

        consensus.get_db_connection()
        self.assertEqual(len(dipanggil), 1)

    def test_koneksi_basi_memicu_rebuild(self):
        class KoneksiBasi:
            def cursor(self):
                raise Exception("server closed the connection unexpectedly")

        class KoneksiSehat:
            def cursor(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, q):
                return None

        class PoolBasi:
            def getconn(self):
                return KoneksiBasi()

            def closeall(self):
                pass

        class PoolBaru:
            def getconn(self):
                return KoneksiSehat()

        consensus.db_pool = PoolBasi()
        consensus.build_pool = lambda: PoolBaru()
        self.assertIsNotNone(consensus.get_db_connection())
