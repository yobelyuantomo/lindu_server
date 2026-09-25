"""Kebijakan aktuator bertingkat berbasis keyakinan model.

Sistem Assignment 2 hanya mengenal dua keadaan: alarm penuh atau diam. Dengan
tersedianya confidence dan lead-time, respons bisa dibuat proporsional terhadap
keyakinan dan urgensi — mengikuti Tabel 5 proposal.

--------------------------------------------------------------------------
Aturan keselamatan yang mengikat
--------------------------------------------------------------------------
1. **Rule-based selalu bisa mencapai tingkat penuh.** Bila konsensus fisika
   P-Wave tercapai, aktuator bekerja penuh tanpa peduli apa kata model. ML
   hanya boleh *menaikkan* respons pada kasus yang rule-based lewatkan, tidak
   pernah *menurunkannya*.

2. **Shadow mode memangkas seluruh wewenang ML.** Saat aktif, tingkat yang
   berasal dari ML diturunkan menjadi SIAGA (tanpa aksi fisik). Yang berasal
   dari konsensus rule-based tetap berjalan penuh.

3. **Menahan aksi bukan berarti diam.** Setiap tingkat di atas NORMAL selalu
   memberi peringatan visual. Yang dibatasi hanya aksi fisik yang mengganggu
   atau sulit dipulihkan (sirine, kunci katup, buka pintu).
"""

#: Tingkat respons, dari paling ringan ke paling berat.
SIAGA = "siaga"
WASPADA = "waspada"
BAHAYA = "bahaya"
KRITIS = "kritis"
NORMAL = "normal"

URUTAN = [NORMAL, SIAGA, WASPADA, BAHAYA, KRITIS]

#: Ambang keyakinan untuk tiap tingkat.
CONF_SIAGA = 0.50
CONF_WASPADA = 0.75
CONF_BAHAYA = 0.90

#: Lead-time (detik) di bawah ini langsung dianggap BAHAYA berapa pun
#: confidence-nya: kalau guncangan kuat memang akan tiba dalam hitungan detik,
#: menunggu keyakinan model bertambah tidak ada gunanya.
LEAD_TIME_KRITIS_S = 5.0

#: Magnitudo prediksi di atas ini menaikkan BAHAYA menjadi KRITIS.
MAGNITUDO_KRITIS = 6.0

#: Aksi fisik per tingkat.
AKSI = {
    NORMAL:  {"led": "hijau",   "siren": False, "lock_valve": False, "open_door": False, "persist_nvs": False},
    SIAGA:   {"led": "kuning",  "siren": False, "lock_valve": False, "open_door": False, "persist_nvs": False},
    WASPADA: {"led": "oranye",  "siren": False, "lock_valve": False, "open_door": False, "persist_nvs": False},
    BAHAYA:  {"led": "merah",   "siren": True,  "lock_valve": True,  "open_door": True,  "persist_nvs": False},
    KRITIS:  {"led": "merah",   "siren": True,  "lock_valve": True,  "open_door": True,  "persist_nvs": True},
}


def _tingkat_dari_ml(confidence, lead_time_s, magnitude_pred):
    """Tingkat yang disarankan model, sebelum aturan keselamatan diterapkan."""
    if lead_time_s is not None and lead_time_s < LEAD_TIME_KRITIS_S:
        return BAHAYA

    if confidence is None:
        return NORMAL
    if confidence >= CONF_BAHAYA:
        if magnitude_pred is not None and magnitude_pred >= MAGNITUDO_KRITIS:
            return KRITIS
        return BAHAYA
    if confidence >= CONF_WASPADA:
        return WASPADA
    if confidence >= CONF_SIAGA:
        return SIAGA
    return NORMAL


def tingkat_tertinggi(a, b):
    return a if URUTAN.index(a) >= URUTAN.index(b) else b


def decide(rule_consensus, ml_result=None, shadow_mode=False):
    """Tentukan tingkat respons dan aksi aktuatornya.

    ``rule_consensus`` adalah hasil validasi fisika P-Wave antar-node — yaitu
    pemicu alarm Assignment 2. Bila True, tingkat minimal adalah BAHAYA.

    ``ml_result`` adalah keluaran ``InferenceEngine.predict()``. Diabaikan bila
    tidak tersedia atau melewati anggaran latensi.
    """
    alasan = []

    tingkat_rule = BAHAYA if rule_consensus else NORMAL
    if rule_consensus:
        alasan.append("konsensus fisika P-Wave tercapai")

    tingkat_ml = NORMAL
    ml_dipakai = bool(
        ml_result
        and ml_result.get("available")
        and not ml_result.get("over_budget")
        and ml_result.get("label") == "earthquake"
    )
    if ml_dipakai:
        tingkat_ml = _tingkat_dari_ml(
            ml_result.get("confidence"),
            ml_result.get("lead_time_pred"),
            ml_result.get("magnitude_pred"),
        )
        if tingkat_ml != NORMAL:
            alasan.append(
                f"ML menilai gempa (confidence {ml_result.get('confidence') or 0:.2f})"
            )
    elif ml_result and ml_result.get("over_budget"):
        alasan.append("hasil ML diabaikan: melewati anggaran latensi")

    # Shadow mode memangkas wewenang ML, bukan wewenang rule-based.
    if shadow_mode and tingkat_ml != NORMAL:
        tingkat_ml = SIAGA
        alasan.append("shadow mode: wewenang ML dibatasi sampai SIAGA")

    tingkat = tingkat_tertinggi(tingkat_rule, tingkat_ml)

    # Penjaga terakhir: konsensus rule-based tidak pernah boleh diturunkan.
    if rule_consensus:
        tingkat = tingkat_tertinggi(tingkat, BAHAYA)

    return {
        "level": tingkat,
        "actions": dict(AKSI[tingkat]),
        "from_rule": tingkat_rule,
        "from_ml": tingkat_ml,
        "shadow_mode": shadow_mode,
        "reason": "; ".join(alasan) or "tidak ada indikasi gempa",
    }


def to_mqtt_level(tingkat):
    """Petakan tingkat ke nilai ``level`` pada payload alarm MQTT.

    Firmware existing hanya mengenal CRITICAL sebagai pemicu sirine penuh.
    Tingkat di bawah BAHAYA sengaja dipetakan ke nilai lain supaya node lama
    yang belum paham tingkat baru tidak ikut membunyikan sirine.
    """
    return {
        NORMAL: "NORMAL",
        SIAGA: "ADVISORY",
        WASPADA: "WATCH",
        BAHAYA: "CRITICAL",
        KRITIS: "CRITICAL",
    }[tingkat]
