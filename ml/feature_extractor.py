"""Ekstraksi fitur dari jendela telemetri seismik.

Modul ini dipakai BERSAMA oleh dua sisi:

- Pelatihan offline (``ml_training/``), saat membangun dataset dari CSV.
- Inferensi runtime (``ml/inference_engine.py``), saat server memproses MQTT.

Keduanya wajib memakai modul yang sama supaya tidak terjadi train/serve skew —
yaitu fitur saat melatih berbeda definisi dengan fitur saat memprediksi.

Sengaja hanya memakai pustaka standar Python (tanpa numpy/pandas) agar jalur
runtime server tidak menanggung dependensi tambahan.

--------------------------------------------------------------------------
CATATAN PENTING: laju telemetri bersifat adaptif
--------------------------------------------------------------------------
Firmware (``SensorManager.cpp``) mengirim telemetri dengan laju yang berbeda
tergantung kuat getaran::

    PGA > 0.12  -> setiap 100 ms  (10 Hz)
    selain itu  -> setiap 1000 ms (1 Hz)

Artinya *kepadatan sampel itu sendiri sudah mengkodekan lapisan pertama filter
rule-based*. Kalau jumlah sampel per jendela dipakai sebagai fitur, model akan
belajar "jendela padat = gempa", yang tidak lain adalah aturan lama yang
disamarkan. Hasil perbandingan ML vs rule-based menjadi tidak bermakna.

Dua konsekuensi yang dipegang modul ini:

1. Jendela dibentuk berdasarkan **durasi waktu**, bukan jumlah baris.
2. Jumlah sampel dan laju sampling **tidak pernah** dijadikan fitur.

Lihat pula ``is_analyzable_window()``: pekerjaan klasifikasi dibatasi pada
jendela yang memang terpicu (ada sampel dengan PGA di atas ambang), supaya
kelas positif dan negatif sama-sama terekam pada 10 Hz.
"""

import math

# ---------------------------------------------------------------------------
# Ambang batas rule-based Assignment 2.
#
# Satu-satunya definisi di seluruh proyek. Jalur runtime (consensus.py) punya
# salinannya sendiri karena alasan historis, tetapi setiap evaluasi baseline
# maupun pembangunan dataset WAJIB memakai konstanta di sini agar
# perbandingan ML vs rule-based memakai aturan yang identik.
# ---------------------------------------------------------------------------
RULE_PGA_MIN = 0.12
RULE_STA_LTA_MIN = 2.0
RULE_FREQ_HZ_MAX = 20.0

#: Panjang jendela analisis dalam detik.
#: Pada 10 Hz (kondisi terpicu) menghasilkan ~20 sampel — cukup untuk statistik
#: yang stabil tanpa membuat jendela lebih panjang dari durasi gelombang-P awal.
WINDOW_SECONDS = 2.0

#: Jumlah sampel minimum agar sebuah jendela layak dihitung fiturnya.
MIN_SAMPLES_PER_WINDOW = 4

_EPS = 1e-9


def rule_based_verdict(pga, sta_lta, freq_hz):
    """Filter tiga lapis Assignment 2.

    Dikembalikan sebagai fungsi agar evaluasi baseline (T2.2) dan pelabelan
    dataset memakai definisi yang persis sama, bukan menyalin ulang ambangnya.

    ``freq_hz`` bernilai ``None`` diperlakukan sebagai TIDAK lolos. Ini berbeda
    dari perilaku lama yang memakai default 0 — nilai 0 lolos ``<= 20`` sehingga
    payload tanpa frekuensi otomatis dianggap gempa.
    """
    if pga is None or sta_lta is None or freq_hz is None:
        return False
    return (
        pga >= RULE_PGA_MIN
        and sta_lta >= RULE_STA_LTA_MIN
        and freq_hz <= RULE_FREQ_HZ_MAX
    )


def is_analyzable_window(rows):
    """Apakah jendela ini termasuk populasi yang dianalisis model?

    Klasifikasi hanya dikerjakan pada jendela yang **terpicu**, yaitu punya
    minimal satu sampel dengan ``pga >= RULE_PGA_MIN``. Alasannya dua:

    - Pada kondisi terpicu firmware mengirim 10 Hz untuk semua jenis getaran,
      sehingga kelas gempa dan non-gempa punya kepadatan sampel yang sama dan
      tidak ada kebocoran laju sampling.
    - Justru di populasi inilah kesalahan sistem lama berada. Getaran tenang
      sudah benar ditolak oleh lapisan pertama dan tidak butuh model.

    Jendela tenang tetap dicatat di database, hanya saja tidak diikutkan ke
    dalam dataset latih. Batasan ini wajib ditulis di bab Limitasi laporan.
    """
    if len(rows) < MIN_SAMPLES_PER_WINDOW:
        return False
    return any((r.get("pga") or 0.0) >= RULE_PGA_MIN for r in rows)


#: Nama fitur, berurutan. Urutan ini adalah kontrak antara pelatihan dan
#: inferensi: ``model_meta.json`` menyimpan salinannya, dan inference engine
#: menolak memuat model bila urutannya tidak cocok.
FEATURE_NAMES = [
    # Amplitudo
    "pga_max",
    "pga_mean",
    "pga_std",
    "pga_crest",
    # Energi
    "energy_cumulative",
    "duration_above_thresh",
    "rise_rate",
    "time_to_peak",
    # Frekuensi
    "freq_mean",
    "freq_min",
    "freq_max",
    "freq_std",
    # Rasio STA/LTA
    "sta_lta_max",
    "sta_lta_mean",
    "sta_lta_std",
    # Geometri getaran
    "hv_ratio",
    "accel_mag_max",
    "accel_mag_mean",
    # Lingkungan
    "pressure_delta",
    "temperature_mean",
    "gas_raw_max",
]


def _stats(values):
    """(mean, std, min, max) yang aman untuk list kosong."""
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std, min(vals), max(vals)


def _timestamps(rows):
    """Deret waktu relatif (detik) terhadap awal jendela."""
    raw = [r.get("sensor_ts") for r in rows]
    if any(t is None for t in raw):
        # Fallback: anggap sampel berjarak seragam. Tidak ideal, tetapi lebih
        # baik daripada membuang jendela — dan hanya terjadi pada data lama.
        return [float(i) / 10.0 for i in range(len(rows))]
    t0 = raw[0]
    return [float(t) - float(t0) for t in raw]


def extract_features(rows):
    """Hitung vektor fitur dari satu jendela telemetri milik SATU node.

    ``rows`` adalah list dict terurut menaik berdasarkan waktu, dengan kunci:
    ``sensor_ts``, ``pga``, ``sta_lta``, ``freq_hz``, ``accel_x``, ``accel_y``,
    ``accel_z``, ``temperature``, ``pressure``, ``gas_raw``.

    Mengembalikan dict dengan kunci persis ``FEATURE_NAMES``. Memanggil dengan
    jendela kosong akan melempar ``ValueError`` — pemanggil bertanggung jawab
    menyaringnya lewat :func:`is_analyzable_window`.
    """
    if not rows:
        raise ValueError("Jendela kosong: tidak ada baris telemetri untuk diolah")

    times = _timestamps(rows)
    pga_vals = [r.get("pga") for r in rows]
    sta_vals = [r.get("sta_lta") for r in rows]
    freq_vals = [r.get("freq_hz") for r in rows]

    pga_clean = [p if p is not None else 0.0 for p in pga_vals]

    pga_mean, pga_std, _, pga_max = _stats(pga_vals)
    freq_mean, freq_std, freq_min, freq_max = _stats(freq_vals)
    sta_mean, sta_std, _, sta_max = _stats(sta_vals)

    # --- Energi kumulatif dan durasi di atas ambang -------------------------
    # Diintegrasikan terhadap waktu (bukan dijumlah per sampel) supaya nilainya
    # tidak ikut berubah saat laju telemetri berpindah antara 1 Hz dan 10 Hz.
    energy = 0.0
    duration_above = 0.0
    for i in range(1, len(rows)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        p = pga_clean[i]
        energy += p * p * dt
        if p >= RULE_PGA_MIN:
            duration_above += dt

    # --- Bentuk amplop getaran ---------------------------------------------
    peak_idx = pga_clean.index(max(pga_clean)) if pga_clean else 0
    time_to_peak = times[peak_idx] - times[0]
    rise = pga_clean[peak_idx] - pga_clean[0]
    rise_rate = rise / (time_to_peak + _EPS) if time_to_peak > 0 else 0.0
    crest = pga_max / (pga_mean + _EPS)

    # --- Geometri getaran ---------------------------------------------------
    # Gempa cenderung didominasi komponen horizontal; hentakan kaki dan benturan
    # meja lebih dominan vertikal. Komponen accel dari firmware sudah bebas
    # gravitasi (dyn_x/y/z), jadi rasio ini bermakna.
    horiz, vert, mags = [], [], []
    for r in rows:
        ax = r.get("accel_x") or 0.0
        ay = r.get("accel_y") or 0.0
        az = r.get("accel_z") or 0.0
        h = math.sqrt(ax * ax + ay * ay)
        horiz.append(h)
        vert.append(abs(az))
        mags.append(math.sqrt(ax * ax + ay * ay + az * az))

    horiz_mean = sum(horiz) / len(horiz)
    vert_mean = sum(vert) / len(vert)
    hv_ratio = horiz_mean / (vert_mean + _EPS)
    mag_mean, _, _, mag_max = _stats(mags)

    # --- Lingkungan ---------------------------------------------------------
    pressures = [r.get("pressure") for r in rows if r.get("pressure") is not None]
    pressure_delta = (pressures[-1] - pressures[0]) if len(pressures) >= 2 else 0.0
    temp_mean, _, _, _ = _stats([r.get("temperature") for r in rows])
    _, _, _, gas_max = _stats([r.get("gas_raw") for r in rows])

    return {
        "pga_max": pga_max,
        "pga_mean": pga_mean,
        "pga_std": pga_std,
        "pga_crest": crest,
        "energy_cumulative": energy,
        "duration_above_thresh": duration_above,
        "rise_rate": rise_rate,
        "time_to_peak": time_to_peak,
        "freq_mean": freq_mean,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "freq_std": freq_std,
        "sta_lta_max": sta_max,
        "sta_lta_mean": sta_mean,
        "sta_lta_std": sta_std,
        "hv_ratio": hv_ratio,
        "accel_mag_max": mag_max,
        "accel_mag_mean": mag_mean,
        "pressure_delta": pressure_delta,
        "temperature_mean": temp_mean,
        "gas_raw_max": gas_max,
    }


def features_to_vector(features):
    """Ubah dict fitur menjadi list terurut sesuai :data:`FEATURE_NAMES`."""
    return [float(features[name]) for name in FEATURE_NAMES]


def build_windows(rows, window_seconds=WINDOW_SECONDS):
    """Potong deret telemetri satu node menjadi jendela non-overlap.

    Pemotongan berdasarkan **durasi**, bukan jumlah baris — lihat catatan laju
    adaptif di docstring modul. Baris diasumsikan sudah terurut menaik menurut
    ``sensor_ts``.
    """
    if not rows:
        return []

    times = _timestamps(rows)
    windows = []
    current = [rows[0]]
    window_start = times[0]

    for i in range(1, len(rows)):
        if times[i] - window_start >= window_seconds:
            windows.append(current)
            current = [rows[i]]
            window_start = times[i]
        else:
            current.append(rows[i])

    if current:
        windows.append(current)
    return windows
