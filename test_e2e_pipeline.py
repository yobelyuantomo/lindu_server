"""Uji end-to-end jalur keputusan, tanpa memerlukan broker maupun database.

``simulate_e2e.py`` menguji sistem lengkap tetapi butuh MQTT dan Postgres
hidup, sehingga tidak bisa dijalankan di CI. Berkas ini menguji rangkaian yang
sama — telemetri masuk, fitur diekstraksi, model memprediksi, keputusan
digabung, aktuator ditentukan — sebagai fungsi murni.

Empat skenario yang diminta T5.2: gempa valid, noise berenergi tinggi,
confidence rendah, dan model tidak tersedia.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.actuator_policy import BAHAYA, NORMAL, SIAGA, decide
from ml.feature_extractor import FEATURE_NAMES, rule_based_verdict  # noqa: E402
from ml.inference_engine import InferenceEngine, combine_verdicts  # noqa: E402


class ModelPalsu:
    """Model tiruan yang menilai dari frekuensi dominan jendela.

    Meniru pola yang ingin dipelajari model sungguhan: gempa berfrekuensi
    rendah dan berkelanjutan, hentakan berfrekuensi tinggi dan impulsif.
    """

    classes_ = ["earthquake", "noise"]

    def __init__(self, confidence=0.95):
        self.confidence = confidence

    def _label(self, vektor):
        freq_mean = vektor[FEATURE_NAMES.index("freq_mean")]
        return "earthquake" if freq_mean <= 20.0 else "noise"

    def predict(self, X):
        return [self._label(v) for v in X]

    def predict_proba(self, X):
        hasil = []
        for v in X:
            if self._label(v) == "earthquake":
                hasil.append([self.confidence, 1 - self.confidence])
            else:
                hasil.append([1 - self.confidence, self.confidence])
        return hasil


def buat_engine(confidence=0.95, shadow_mode=False):
    engine = InferenceEngine(shadow_mode=shadow_mode)
    engine._model = ModelPalsu(confidence)
    engine._meta = {"model_version": "e2e", "feature_names": list(FEATURE_NAMES)}
    engine._load_error = None
    return engine


def alirkan(engine, pga, sta_lta, freq_hz, n=20, node_id="NODE_A"):
    """Kirim n pesan telemetri @10 Hz, kembalikan hasil prediksi terakhir."""
    hasil = None
    for i in range(n):
        hasil = engine.predict({
            "node_id": node_id, "ts": i * 0.1, "pga": pga,
            "sta_lta": sta_lta, "freq_hz": freq_hz,
            "ax": pga * 0.8, "ay": pga * 0.4, "az": pga * 0.1,
            "temperature": 28.0, "pressure": 1010.0, "gas_raw": 100,
        })
    return hasil


class TestSkenarioGempaValid(unittest.TestCase):
    """Guncangan kuat, berkelanjutan, frekuensi rendah — gempa sungguhan."""

    def test_rule_dan_ml_sepakat_aktuator_penuh(self):
        engine = buat_engine()
        hasil = alirkan(engine, pga=0.6, sta_lta=4.0, freq_hz=6.0)

        self.assertTrue(hasil.available)
        self.assertEqual(hasil["label"], "earthquake")

        rule = rule_based_verdict(0.6, 4.0, 6.0)
        self.assertTrue(rule)

        keputusan = combine_verdicts(rule, hasil)
        self.assertEqual(keputusan["decision"], "confirmed")

        aksi = decide(rule_consensus=True, ml_result=hasil)
        self.assertEqual(aksi["level"], BAHAYA)
        self.assertTrue(aksi["actions"]["siren"])
        self.assertTrue(aksi["actions"]["lock_valve"])


class TestSkenarioNoiseBerenergiTinggi(unittest.TestCase):
    """Hentakan keras: PGA dan STA/LTA lolos, tapi frekuensinya tinggi.

    Ini yang selama ini dibuang oleh lapisan ketiga filter. ML harus setuju.
    """

    def test_keduanya_menolak(self):
        engine = buat_engine()
        hasil = alirkan(engine, pga=0.9, sta_lta=5.0, freq_hz=35.0)

        self.assertEqual(hasil["label"], "noise")
        rule = rule_based_verdict(0.9, 5.0, 35.0)
        self.assertFalse(rule, "frekuensi 35 Hz harus gagal di lapisan ketiga")

        keputusan = combine_verdicts(rule, hasil)
        self.assertEqual(keputusan["decision"], "normal")
        self.assertFalse(keputusan["alert"])

        aksi = decide(rule_consensus=False, ml_result=hasil)
        self.assertEqual(aksi["level"], NORMAL)
        self.assertFalse(aksi["actions"]["siren"])


class TestSkenarioConfidenceRendah(unittest.TestCase):
    """Rule lolos tetapi model ragu — aksi ditahan, peringatan tetap ada."""

    def test_aktuator_ditahan_tapi_tidak_diam(self):
        engine = buat_engine(confidence=0.55)
        hasil = alirkan(engine, pga=0.5, sta_lta=3.0, freq_hz=9.0)

        rule = rule_based_verdict(0.5, 3.0, 9.0)
        self.assertTrue(rule)

        keputusan = combine_verdicts(rule, hasil, confidence_min=0.75)
        self.assertEqual(keputusan["decision"], "suspected_false_positive")
        self.assertFalse(keputusan["trigger_actuator"])
        self.assertTrue(keputusan["alert"], "peringatan lokal harus tetap muncul")

    def test_konsensus_fisika_tetap_memicu_penuh(self):
        """Ragu-ragunya model tidak boleh melemahkan konsensus antar-node."""
        engine = buat_engine(confidence=0.55)
        hasil = alirkan(engine, pga=0.5, sta_lta=3.0, freq_hz=9.0)

        aksi = decide(rule_consensus=True, ml_result=hasil)
        self.assertEqual(aksi["level"], BAHAYA)
        self.assertTrue(aksi["actions"]["siren"])


class TestSkenarioModelTidakTersedia(unittest.TestCase):
    """Sistem harus berperilaku persis seperti Assignment 2."""

    def test_tanpa_model_perilaku_assignment_2(self):
        engine = InferenceEngine(models_dir="/path/yang/tidak/ada")
        engine.load()
        hasil = alirkan(engine, pga=0.6, sta_lta=4.0, freq_hz=6.0)
        self.assertFalse(hasil.available)

        rule = rule_based_verdict(0.6, 4.0, 6.0)
        keputusan = combine_verdicts(rule, hasil)
        self.assertEqual(keputusan["decision"], "rule_only")
        self.assertTrue(keputusan["trigger_actuator"])

        aksi = decide(rule_consensus=True, ml_result=hasil)
        self.assertEqual(aksi["level"], BAHAYA)
        self.assertTrue(aksi["actions"]["siren"])

    def test_tanpa_model_noise_tetap_ditolak(self):
        engine = InferenceEngine(models_dir="/path/yang/tidak/ada")
        engine.load()
        hasil = alirkan(engine, pga=0.9, sta_lta=5.0, freq_hz=35.0)

        rule = rule_based_verdict(0.9, 5.0, 35.0)
        aksi = decide(rule_consensus=rule, ml_result=hasil)
        self.assertEqual(aksi["level"], NORMAL)


class TestShadowModeEndToEnd(unittest.TestCase):
    def test_ml_tidak_menyentuh_aktuator(self):
        engine = buat_engine(shadow_mode=True)
        hasil = alirkan(engine, pga=0.6, sta_lta=4.0, freq_hz=6.0)
        self.assertTrue(hasil["shadow_mode"])

        # Rule-based tidak lolos (mis. baru satu node), jadi wewenang hanya ML.
        aksi = decide(rule_consensus=False, ml_result=hasil, shadow_mode=True)
        self.assertEqual(aksi["level"], SIAGA)
        self.assertFalse(aksi["actions"]["siren"])
        self.assertFalse(aksi["actions"]["lock_valve"])


class TestTelemetriTanpaFreqHz(unittest.TestCase):
    def test_ditolak_filter_rule_based(self):
        """Regresi: payload tanpa freq_hz dulu otomatis dianggap gempa."""
        self.assertFalse(rule_based_verdict(0.9, 5.0, None))


if __name__ == "__main__":
    unittest.main()
