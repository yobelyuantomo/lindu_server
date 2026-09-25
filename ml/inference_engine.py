"""Mesin inferensi ML untuk jalur deteksi Lindu-EEW.

Modul ini disisipkan di antara penerimaan telemetri MQTT dan mesin konsensus.
Ia menyimpan jendela telemetri per node, menjalankan model klasifikasi, dan
mengembalikan hasil prediksi beserta tingkat keyakinannya.

--------------------------------------------------------------------------
Prinsip yang tidak boleh dilanggar
--------------------------------------------------------------------------
**ML tidak pernah menjadi titik kegagalan tunggal.** Jalur rule-based
Assignment 2 tetap berjalan penuh dan independen. Setiap kegagalan di sini —
model tidak ada, metadata rusak, urutan fitur tidak cocok, inferensi terlalu
lama, atau pengecualian tak terduga — berakhir pada hasil ``available=False``
dan sistem melanjutkan persis seperti sebelum ada ML.

**Tidak pernah melempar pengecualian ke pemanggil.** ``predict()`` menangkap
semuanya. Modul yang memicu sirine tidak boleh bisa dirobohkan oleh bug di
lapisan kecerdasan.

**Anggaran latensi ditegakkan.** Inferensi yang melewati anggaran tetap
dikembalikan tetapi ditandai ``over_budget``, sehingga pemanggil bisa
mengabaikannya. Peringatan dini yang terlambat tidak ada gunanya.
"""

import json
import os
import time
from collections import deque

from ml.feature_extractor import (
    FEATURE_NAMES,
    MIN_SAMPLES_PER_WINDOW,
    WINDOW_SECONDS,
    extract_features,
    features_to_vector,
    is_analyzable_window,
    rule_based_verdict,
)

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "classifier.joblib"
META_FILENAME = "model_meta.json"

#: Anggaran latensi inferensi. Di atas ini hasil ditandai over_budget.
LATENCY_BUDGET_MS = 50.0

#: Batas jumlah baris yang disimpan per node. Pada 10 Hz, 200 baris = 20 detik —
#: jauh lebih dari cukup untuk jendela 2 detik, sekaligus memberi batas atas
#: tegas pada pemakaian memori saat node bertambah banyak.
MAX_ROWS_PER_NODE = 200


class InferenceResult(dict):
    """Hasil prediksi. Turunan dict agar mudah diserialisasi ke database."""

    @property
    def available(self):
        return self.get("available", False)


def _unavailable(reason, **extra):
    hasil = InferenceResult(
        available=False,
        reason=reason,
        label=None,
        confidence=None,
        model_version=None,
        latency_ms=None,
        over_budget=False,
    )
    hasil.update(extra)
    return hasil


class InferenceEngine:
    """Menyimpan jendela per node dan menjalankan model klasifikasi."""

    def __init__(self, models_dir=DEFAULT_MODELS_DIR, window_seconds=WINDOW_SECONDS,
                 latency_budget_ms=LATENCY_BUDGET_MS, shadow_mode=False):
        self.models_dir = models_dir
        self.window_seconds = window_seconds
        self.latency_budget_ms = latency_budget_ms
        self.shadow_mode = shadow_mode

        self._buffers = {}
        self._model = None
        self._meta = None
        self._load_error = None
        self._stats = {"predictions": 0, "fallbacks": 0, "over_budget": 0}

    # -- pemuatan model ----------------------------------------------------

    def load(self):
        """Muat model dan metadata. Mengembalikan True bila berhasil.

        Kegagalan tidak dilempar — dicatat dan membuat mesin beroperasi dalam
        mode tidak tersedia, sehingga sistem otomatis kembali ke rule-based.
        """
        model_path = os.path.join(self.models_dir, MODEL_FILENAME)
        meta_path = os.path.join(self.models_dir, META_FILENAME)

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            self._load_error = "berkas model atau metadata tidak ditemukan"
            return False

        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError) as exc:
            self._load_error = f"metadata tidak terbaca: {exc}"
            return False

        # Kontrak urutan fitur. Ini penjaga utama terhadap train/serve skew:
        # model yang dilatih dengan urutan fitur berbeda akan menghasilkan
        # prediksi yang tampak masuk akal tetapi sepenuhnya salah, dan tidak
        # ada gejala apa pun yang menandainya.
        fitur_model = meta.get("feature_names")
        if fitur_model != list(FEATURE_NAMES):
            self._load_error = (
                "urutan fitur model tidak cocok dengan feature_extractor "
                f"(model: {len(fitur_model or [])} fitur, "
                f"runtime: {len(FEATURE_NAMES)} fitur)"
            )
            return False

        try:
            import joblib

            self._model = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - kegagalan apa pun = fallback
            self._load_error = f"model gagal dimuat: {exc}"
            self._model = None
            return False

        self._meta = meta
        self._load_error = None
        return True

    @property
    def ready(self):
        return self._model is not None

    @property
    def model_version(self):
        return (self._meta or {}).get("model_version")

    @property
    def load_error(self):
        return self._load_error

    # -- buffer per node ---------------------------------------------------

    def add_telemetry(self, payload):
        """Masukkan satu payload telemetri ke buffer node terkait.

        Payload MQTT memakai kunci ``ax``/``ay``/``az`` sementara fitur memakai
        ``accel_x``/``accel_y``/``accel_z``; pemetaan dilakukan di sini supaya
        pemanggil tidak perlu tahu perbedaan itu.
        """
        node_id = payload.get("node_id")
        if not node_id:
            return None

        row = {
            "sensor_ts": payload.get("ts"),
            "pga": payload.get("pga"),
            "sta_lta": payload.get("sta_lta"),
            "freq_hz": payload.get("freq_hz"),
            "accel_x": payload.get("ax", payload.get("accel_x")),
            "accel_y": payload.get("ay", payload.get("accel_y")),
            "accel_z": payload.get("az", payload.get("accel_z")),
            "temperature": payload.get("temperature"),
            "pressure": payload.get("pressure"),
            "gas_raw": payload.get("gas_raw"),
        }
        buffer = self._buffers.setdefault(node_id, deque(maxlen=MAX_ROWS_PER_NODE))
        buffer.append(row)
        return row

    def current_window(self, node_id):
        """Ambil baris dalam ``window_seconds`` terakhir milik satu node."""
        buffer = self._buffers.get(node_id)
        if not buffer:
            return []

        rows = list(buffer)
        terakhir = rows[-1].get("sensor_ts")
        if terakhir is None:
            return rows[-MIN_SAMPLES_PER_WINDOW * 5:]

        batas = terakhir - self.window_seconds
        return [r for r in rows if r.get("sensor_ts") is not None and r["sensor_ts"] >= batas]

    def forget_node(self, node_id):
        self._buffers.pop(node_id, None)

    # -- inferensi ---------------------------------------------------------

    def predict(self, payload):
        """Masukkan telemetri, lalu prediksi bila jendelanya memadai.

        Tidak pernah melempar pengecualian. Kegagalan apa pun menghasilkan
        ``available=False`` dan pemanggil melanjutkan dengan rule-based.
        """
        try:
            return self._predict(payload)
        except Exception as exc:  # noqa: BLE001 - lihat docstring modul
            self._stats["fallbacks"] += 1
            return _unavailable(f"pengecualian tak terduga: {exc}")

    def _predict(self, payload):
        self.add_telemetry(payload)

        if not self.ready:
            self._stats["fallbacks"] += 1
            return _unavailable(self._load_error or "model belum dimuat")

        node_id = payload.get("node_id")
        window = self.current_window(node_id)
        if not is_analyzable_window(window):
            return _unavailable(
                "jendela belum memadai",
                n_samples=len(window),
            )

        mulai = time.perf_counter()
        fitur = extract_features(window)
        vektor = features_to_vector(fitur)
        label, confidence = self._run_model(vektor)
        latency_ms = (time.perf_counter() - mulai) * 1000.0

        over_budget = latency_ms > self.latency_budget_ms
        self._stats["predictions"] += 1
        if over_budget:
            self._stats["over_budget"] += 1

        return InferenceResult(
            available=True,
            reason=None,
            node_id=node_id,
            label=label,
            confidence=confidence,
            features=fitur,
            model_version=self.model_version,
            latency_ms=latency_ms,
            over_budget=over_budget,
            shadow_mode=self.shadow_mode,
            n_samples=len(window),
        )

    def _run_model(self, vektor):
        """Jalankan model, kembalikan (label, confidence)."""
        prediksi = self._model.predict([vektor])
        label = prediksi[0]

        confidence = None
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba([vektor])[0]
            kelas = list(getattr(self._model, "classes_", []))
            if kelas and label in kelas:
                confidence = float(proba[kelas.index(label)])
            elif len(proba):
                confidence = float(max(proba))
        return label, confidence

    # -- diagnostik --------------------------------------------------------

    def stats(self):
        return {
            **self._stats,
            "ready": self.ready,
            "model_version": self.model_version,
            "shadow_mode": self.shadow_mode,
            "load_error": self._load_error,
            "nodes_buffered": len(self._buffers),
        }


def combine_verdicts(rule_passed, ml_result, confidence_min=0.75):
    """Gabungkan verdict rule-based dan ML menjadi satu keputusan.

    Mengikuti Tabel 4 proposal. Yang perlu dipegang: **ML tidak pernah boleh
    membatalkan alarm rule-based.** Bila rule lolos tetapi ML menyanggah,
    sistem menahan aksi aktuator namun tetap memunculkan peringatan lokal dan
    mencatat kejadiannya. Sistem tidak pernah diam total pada guncangan yang
    sudah lolos filter fisika.
    """
    ml_ada = bool(ml_result and ml_result.get("available") and not ml_result.get("over_budget"))
    ml_label = ml_result.get("label") if ml_ada else None
    ml_conf = ml_result.get("confidence") if ml_ada else None
    ml_yakin_gempa = ml_ada and ml_label == "earthquake" and (ml_conf or 0.0) >= confidence_min

    if not ml_ada:
        # Tanpa ML, perilaku persis seperti Assignment 2.
        return {
            "decision": "rule_only",
            "trigger_actuator": rule_passed,
            "alert": rule_passed,
            "reason": "ML tidak tersedia; memakai keputusan rule-based",
        }

    if rule_passed and ml_yakin_gempa:
        return {
            "decision": "confirmed",
            "trigger_actuator": True,
            "alert": True,
            "reason": "rule-based dan ML sepakat gempa",
        }

    if rule_passed and not ml_yakin_gempa:
        return {
            "decision": "suspected_false_positive",
            "trigger_actuator": False,
            "alert": True,
            "reason": f"rule-based lolos tetapi ML menilai '{ml_label}'; "
                      "aksi aktuator ditahan, peringatan lokal tetap muncul",
        }

    if not rule_passed and ml_yakin_gempa:
        return {
            "decision": "ml_early_detection",
            "trigger_actuator": False,
            "alert": True,
            "reason": "ML mendeteksi gempa sebelum rule-based; menunggu konfirmasi node lain",
        }

    return {
        "decision": "normal",
        "trigger_actuator": False,
        "alert": False,
        "reason": "rule-based dan ML sepakat bukan gempa",
    }


__all__ = [
    "InferenceEngine",
    "InferenceResult",
    "combine_verdicts",
    "rule_based_verdict",
    "LATENCY_BUDGET_MS",
    "MAX_ROWS_PER_NODE",
]
