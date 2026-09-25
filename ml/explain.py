"""Penjelasan keputusan model (SHAP) dan penilaian kesehatan sensor.

--------------------------------------------------------------------------
Kenapa penjelasan itu perlu
--------------------------------------------------------------------------
Sistem ini mengunci katup gas dan membuka pintu darurat. Keputusan seperti itu
harus bisa diaudit: ketika sirine berbunyi pukul tiga pagi, seseorang akan
bertanya kenapa, dan "modelnya bilang begitu" bukan jawaban yang bisa
dipertanggungjawabkan.

SHAP mahal — jauh lebih mahal daripada inferensinya sendiri. Karena itu ia
**tidak pernah** dijalankan pada setiap telemetri, melainkan hanya pada
keputusan yang benar-benar berdampak (tingkat BAHAYA dan KRITIS). Pada laju
10 Hz per node, menghitungnya setiap saat akan menghabiskan anggaran latensi
berkali-kali lipat.

Modul ini tidak pernah melempar. Penjelasan adalah tambahan; kegagalan
menghasilkannya tidak boleh menjatuhkan jalur deteksi.
"""

from ml.feature_extractor import FEATURE_NAMES

#: Berapa banyak fitur teratas yang disimpan sebagai alasan.
TOP_N = 5

#: Skor anomali di atas ini dianggap di luar kebiasaan.
#: Diisi dari metadata model (ambang_p95 per node) bila tersedia.
DEFAULT_ANOMALY_THRESHOLD = 0.6

#: Berapa lama anomali harus bertahan sebelum dicurigai sebagai masalah
#: sensor, bukan kejadian seismik. Gempa berlangsung puluhan detik; sensor
#: yang bergeser atau rusak menghasilkan anomali yang tidak berhenti.
SENSOR_FAULT_SECONDS = 300.0


class Explainer:
    """Pembungkus SHAP yang aman dipakai di jalur runtime."""

    def __init__(self, model):
        self._explainer = None
        self._np = None
        self._error = None
        try:
            import numpy as np
            import shap

            self._explainer = shap.TreeExplainer(model)
            # shap menuntut masukan bertipe array numpy; list biasa ditolak
            # dengan AttributeError di dalam validasinya. Referensinya disimpan
            # di sini supaya jalur runtime tidak mengimpor numpy per panggilan.
            self._np = np
        except Exception as exc:  # noqa: BLE001 - penjelasan bersifat tambahan
            self._error = f"SHAP tidak tersedia: {exc}"

    @property
    def ready(self):
        return self._explainer is not None

    @property
    def error(self):
        return self._error

    def explain(self, vektor, class_index=0, top_n=TOP_N):
        """Kontribusi fitur terbesar terhadap satu keputusan.

        Mengembalikan list ``{feature, contribution, direction}`` terurut
        menurun berdasarkan besar kontribusinya, atau ``[]`` bila gagal.
        """
        if not self.ready:
            return []
        if len(vektor) != len(FEATURE_NAMES):
            return []
        try:
            nilai = self._explainer.shap_values(
                self._np.asarray([vektor], dtype=float)
            )
            kontribusi = _ambil_baris(nilai, class_index)
            if kontribusi is None or len(kontribusi) != len(FEATURE_NAMES):
                return []

            pasangan = [
                {
                    "feature": nama,
                    "contribution": float(v),
                    "direction": "mendukung" if v > 0 else "menyanggah",
                }
                for nama, v in zip(FEATURE_NAMES, kontribusi)
            ]
            pasangan.sort(key=lambda p: -abs(p["contribution"]))
            return pasangan[:top_n]
        except Exception:  # noqa: BLE001
            return []


def _ambil_baris(nilai, class_index):
    """Ambil vektor kontribusi satu sampel dari keluaran SHAP.

    Bentuk keluarannya berbeda antar versi shap: bisa list per kelas, array
    3 dimensi (sampel, fitur, kelas), atau array 2 dimensi untuk model biner.
    Ketiganya ditangani supaya tidak terikat pada satu versi.
    """
    try:
        if isinstance(nilai, list):
            arr = nilai[class_index] if class_index < len(nilai) else nilai[0]
            return list(arr[0])

        bentuk = getattr(nilai, "shape", None)
        if bentuk is None:
            return None
        if len(bentuk) == 3:
            return list(nilai[0, :, class_index])
        if len(bentuk) == 2:
            return list(nilai[0])
    except Exception:  # noqa: BLE001
        return None
    return None


def ringkas_alasan(kontribusi):
    """Ubah kontribusi menjadi satu kalimat untuk dashboard."""
    if not kontribusi:
        return "penjelasan tidak tersedia"
    bagian = [
        f"{k['feature']} ({'+' if k['contribution'] > 0 else ''}{k['contribution']:.3f})"
        for k in kontribusi[:3]
    ]
    return "didominasi oleh " + ", ".join(bagian)


class SensorHealthTracker:
    """Membedakan anomali seismik dari sensor yang bermasalah.

    Skor anomali yang tinggi saja tidak memberi tahu apa pun tentang
    penyebabnya. Yang membedakan adalah **berapa lama ia bertahan**: gempa
    berlangsung puluhan detik, sedangkan sensor yang bergeser atau mulai rusak
    menghasilkan anomali yang tidak berhenti.

    Penting untuk tidak salah menyimpulkan ke arah sebaliknya — menganggap
    gempa sungguhan sebagai sensor rusak akan membungkam peringatan. Karena
    itu ambang waktunya dibuat panjang (5 menit), jauh melampaui durasi
    guncangan gempa mana pun.
    """

    def __init__(self, threshold=DEFAULT_ANOMALY_THRESHOLD,
                 fault_seconds=SENSOR_FAULT_SECONDS):
        self.threshold = threshold
        self.fault_seconds = fault_seconds
        self._anomali_sejak = {}

    def update(self, node_id, anomaly_score, now):
        """Perbarui keadaan satu node, kembalikan penilaiannya."""
        if anomaly_score is None:
            return self._hasil(node_id, "tidak diketahui", None)

        if anomaly_score < self.threshold:
            self._anomali_sejak.pop(node_id, None)
            return self._hasil(node_id, "normal", 0.0)

        mulai = self._anomali_sejak.setdefault(node_id, now)
        durasi = now - mulai
        status = "dugaan sensor bermasalah" if durasi >= self.fault_seconds else "anomali"
        return self._hasil(node_id, status, durasi)

    def _hasil(self, node_id, status, durasi):
        return {
            "node_id": node_id,
            "status": status,
            "anomalous_for_s": durasi,
            "is_sensor_fault": status == "dugaan sensor bermasalah",
            "recommendation": (
                "Periksa pemasangan fisik sensor node ini. Anomali yang bertahan "
                f"lebih dari {int(self.fault_seconds / 60)} menit jauh melampaui "
                "durasi guncangan gempa mana pun."
                if status == "dugaan sensor bermasalah" else ""
            ),
        }

    def reset(self, node_id=None):
        if node_id is None:
            self._anomali_sejak.clear()
        else:
            self._anomali_sejak.pop(node_id, None)
