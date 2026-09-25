"""Uji ml/inference_engine.py.

Fokus utamanya bukan akurasi model, melainkan **perilaku saat gagal**: mesin
ini menyentuh jalur yang memicu sirine, jadi yang paling penting dibuktikan
adalah bahwa tidak ada kegagalan di lapisan ML yang bisa merobohkan atau
membungkam deteksi rule-based.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402
from ml.inference_engine import (  # noqa: E402
    MAX_ROWS_PER_NODE,
    META_FILENAME,
    InferenceEngine,
    combine_verdicts,
)


class ModelPalsu:
    """Model tiruan agar mesin bisa diuji tanpa scikit-learn."""

    classes_ = ["earthquake", "noise"]

    def __init__(self, label="earthquake", proba=(0.9, 0.1), meledak=False):
        self._label = label
        self._proba = proba
        self._meledak = meledak

    def predict(self, X):
        if self._meledak:
            raise RuntimeError("model rusak")
        return [self._label] * len(X)

    def predict_proba(self, X):
        return [list(self._proba)] * len(X)


def payload(node_id="NODE_A", ts=0.0, pga=0.5, sta_lta=3.0, freq_hz=8.0):
    return {
        "node_id": node_id, "ts": ts, "pga": pga, "sta_lta": sta_lta,
        "freq_hz": freq_hz, "ax": 0.4, "ay": 0.3, "az": 0.1,
        "temperature": 28.0, "pressure": 1010.0, "gas_raw": 100,
    }


def isi_jendela(engine, node_id="NODE_A", n=20, pga=0.5, mulai=0.0):
    """Isi buffer dengan n sampel @10 Hz; kembalikan payload terakhir."""
    terakhir = None
    for i in range(n):
        terakhir = payload(node_id=node_id, ts=mulai + i * 0.1, pga=pga)
        engine.add_telemetry(terakhir)
    return terakhir


def pasang_model(engine, model):
    engine._model = model
    engine._meta = {"model_version": "uji-1", "feature_names": list(FEATURE_NAMES)}
    engine._load_error = None


class TestPemuatanModel(unittest.TestCase):
    def test_berkas_tidak_ada_tidak_melempar(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = InferenceEngine(models_dir=tmp)
            self.assertFalse(engine.load())
            self.assertFalse(engine.ready)
            self.assertIn("tidak ditemukan", engine.load_error)

    def test_metadata_rusak_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "classifier.joblib"), "w").close()
            with open(os.path.join(tmp, META_FILENAME), "w", encoding="utf-8") as h:
                h.write("{ bukan json")
            engine = InferenceEngine(models_dir=tmp)
            self.assertFalse(engine.load())
            self.assertIn("tidak terbaca", engine.load_error)

    def test_urutan_fitur_tidak_cocok_ditolak(self):
        """Penjaga train/serve skew.

        Model dengan urutan fitur berbeda menghasilkan prediksi yang tampak
        masuk akal tetapi sepenuhnya salah, tanpa gejala apa pun. Harus
        ditolak saat memuat, bukan dibiarkan berjalan.
        """
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "classifier.joblib"), "w").close()
            with open(os.path.join(tmp, META_FILENAME), "w", encoding="utf-8") as h:
                json.dump({"feature_names": ["hanya", "tiga", "fitur"]}, h)
            engine = InferenceEngine(models_dir=tmp)
            self.assertFalse(engine.load())
            self.assertIn("tidak cocok", engine.load_error)

    def test_urutan_fitur_terbalik_juga_ditolak(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "classifier.joblib"), "w").close()
            with open(os.path.join(tmp, META_FILENAME), "w", encoding="utf-8") as h:
                json.dump({"feature_names": list(reversed(FEATURE_NAMES))}, h)
            engine = InferenceEngine(models_dir=tmp)
            self.assertFalse(engine.load())


class TestBuffer(unittest.TestCase):
    def test_buffer_dibatasi(self):
        """Tanpa batas, node yang lama hidup akan menghabiskan memori."""
        engine = InferenceEngine()
        isi_jendela(engine, n=MAX_ROWS_PER_NODE + 150)
        self.assertEqual(len(engine._buffers["NODE_A"]), MAX_ROWS_PER_NODE)

    def test_node_dipisah(self):
        engine = InferenceEngine()
        isi_jendela(engine, node_id="NODE_A", n=5)
        isi_jendela(engine, node_id="NODE_B", n=3)
        self.assertEqual(len(engine._buffers["NODE_A"]), 5)
        self.assertEqual(len(engine._buffers["NODE_B"]), 3)

    def test_jendela_dipotong_per_durasi(self):
        engine = InferenceEngine(window_seconds=2.0)
        isi_jendela(engine, n=60)  # 6 detik @10 Hz
        self.assertLessEqual(len(engine.current_window("NODE_A")), 21)

    def test_payload_tanpa_node_id_diabaikan(self):
        engine = InferenceEngine()
        self.assertIsNone(engine.add_telemetry({"pga": 0.5}))
        self.assertEqual(engine._buffers, {})

    def test_pemetaan_ax_ke_accel_x(self):
        engine = InferenceEngine()
        row = engine.add_telemetry(payload())
        self.assertEqual(row["accel_x"], 0.4)
        self.assertEqual(row["accel_z"], 0.1)

    def test_forget_node(self):
        engine = InferenceEngine()
        isi_jendela(engine, n=5)
        engine.forget_node("NODE_A")
        self.assertEqual(engine.current_window("NODE_A"), [])


class TestPredict(unittest.TestCase):
    def test_model_belum_dimuat_menghasilkan_unavailable(self):
        engine = InferenceEngine()
        hasil = engine.predict(payload())
        self.assertFalse(hasil.available)
        self.assertIsNone(hasil["label"])

    def test_jendela_belum_memadai(self):
        engine = InferenceEngine()
        pasang_model(engine, ModelPalsu())
        hasil = engine.predict(payload())
        self.assertFalse(hasil.available)
        self.assertIn("belum memadai", hasil["reason"])

    def test_prediksi_berhasil(self):
        engine = InferenceEngine()
        pasang_model(engine, ModelPalsu(label="earthquake", proba=(0.9, 0.1)))
        isi_jendela(engine, n=19)
        hasil = engine.predict(payload(ts=1.9))
        self.assertTrue(hasil.available)
        self.assertEqual(hasil["label"], "earthquake")
        self.assertAlmostEqual(hasil["confidence"], 0.9)
        self.assertEqual(hasil["model_version"], "uji-1")
        self.assertIsNotNone(hasil["latency_ms"])

    def test_model_meledak_tidak_merobohkan_pemanggil(self):
        """Kegagalan di lapisan ML tidak boleh menjatuhkan jalur alarm."""
        engine = InferenceEngine()
        pasang_model(engine, ModelPalsu(meledak=True))
        isi_jendela(engine, n=19)
        hasil = engine.predict(payload(ts=1.9))
        self.assertFalse(hasil.available)
        self.assertIn("pengecualian", hasil["reason"])

    def test_jendela_tidak_terpicu_tidak_diprediksi(self):
        engine = InferenceEngine()
        pasang_model(engine, ModelPalsu())
        isi_jendela(engine, n=19, pga=0.01)
        hasil = engine.predict(payload(ts=1.9, pga=0.01))
        self.assertFalse(hasil.available)

    def test_anggaran_latensi_ditandai(self):
        engine = InferenceEngine(latency_budget_ms=-1.0)  # paksa selalu lewat
        pasang_model(engine, ModelPalsu())
        isi_jendela(engine, n=19)
        hasil = engine.predict(payload(ts=1.9))
        self.assertTrue(hasil["over_budget"])
        self.assertEqual(engine.stats()["over_budget"], 1)


class TestCombineVerdicts(unittest.TestCase):
    def _ml(self, label="earthquake", confidence=0.9, available=True, over_budget=False):
        return {"available": available, "label": label,
                "confidence": confidence, "over_budget": over_budget}

    def test_keduanya_sepakat_gempa(self):
        hasil = combine_verdicts(True, self._ml())
        self.assertEqual(hasil["decision"], "confirmed")
        self.assertTrue(hasil["trigger_actuator"])
        self.assertTrue(hasil["alert"])

    def test_rule_lolos_ml_menyanggah_tetap_memberi_peringatan(self):
        """ML tidak boleh MEMBUNGKAM alarm rule-based, hanya menahan aktuator."""
        hasil = combine_verdicts(True, self._ml(label="noise", confidence=0.95))
        self.assertEqual(hasil["decision"], "suspected_false_positive")
        self.assertFalse(hasil["trigger_actuator"])
        self.assertTrue(hasil["alert"], "peringatan lokal harus tetap muncul")

    def test_confidence_rendah_diperlakukan_seperti_menyanggah(self):
        hasil = combine_verdicts(True, self._ml(confidence=0.4))
        self.assertEqual(hasil["decision"], "suspected_false_positive")
        self.assertTrue(hasil["alert"])

    def test_ml_mendeteksi_lebih_dulu(self):
        hasil = combine_verdicts(False, self._ml())
        self.assertEqual(hasil["decision"], "ml_early_detection")
        self.assertFalse(hasil["trigger_actuator"])
        self.assertTrue(hasil["alert"])

    def test_keduanya_normal(self):
        hasil = combine_verdicts(False, self._ml(label="noise"))
        self.assertEqual(hasil["decision"], "normal")
        self.assertFalse(hasil["alert"])

    def test_ml_tidak_tersedia_memakai_rule_saja(self):
        for ml in (None, self._ml(available=False), self._ml(over_budget=True)):
            hasil = combine_verdicts(True, ml)
            self.assertEqual(hasil["decision"], "rule_only")
            self.assertTrue(
                hasil["trigger_actuator"],
                "tanpa ML, perilaku harus persis seperti Assignment 2",
            )

    def test_ml_tidak_tersedia_dan_rule_gagal(self):
        hasil = combine_verdicts(False, None)
        self.assertEqual(hasil["decision"], "rule_only")
        self.assertFalse(hasil["trigger_actuator"])


if __name__ == "__main__":
    unittest.main()
