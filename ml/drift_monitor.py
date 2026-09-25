"""Pemantauan pergeseran distribusi (drift) fitur masukan model.

Model dilatih pada kondisi lapangan tertentu. Bila kondisi itu berubah — node
dipindah, sensor dipasang ulang, ada konstruksi baru di sebelah — distribusi
fitur ikut bergeser dan akurasi model bisa turun **tanpa gejala apa pun**.
Prediksi tetap keluar, confidence tetap tinggi, dan tidak ada yang tahu bahwa
model sudah tidak relevan.

Dipakai ukuran Population Stability Index (PSI), yang membandingkan distribusi
sekarang terhadap distribusi saat pelatihan per-bin::

    PSI = sum( (p_sekarang - p_latih) * ln(p_sekarang / p_latih) )

Ambang yang lazim dipakai di industri:

    PSI < 0.10   stabil
    0.10 - 0.25  bergeser, perlu diperhatikan
    PSI > 0.25   bergeser signifikan, pertimbangkan melatih ulang

Sengaja tanpa numpy/scipy agar bisa berjalan di runtime server yang memang
tidak memasang keduanya.
"""

import math

PSI_STABIL = 0.10
PSI_SIGNIFIKAN = 0.25

#: Jumlah bin saat membangun histogram baseline.
N_BINS = 10

#: Jumlah sampel minimum sebelum PSI layak dihitung. Di bawah ini, nilai PSI
#: lebih mencerminkan kebetulan daripada pergeseran nyata.
MIN_SAMPLES = 50

_EPS = 1e-6


def build_baseline(values, n_bins=N_BINS):
    """Bangun histogram baseline dari nilai-nilai saat pelatihan.

    Mengembalikan dict berisi tepi bin dan proporsi tiap bin. Disimpan ke
    ``model_meta.json`` agar runtime tidak perlu memuat ulang dataset latih.
    """
    bersih = sorted(v for v in values if v is not None)
    if len(bersih) < 2:
        return None

    lo, hi = bersih[0], bersih[-1]
    if hi - lo < _EPS:
        # Fitur konstan: tidak ada yang bisa bergeser.
        return {"constant": lo, "edges": None, "proportions": None}

    lebar = (hi - lo) / n_bins
    edges = [lo + i * lebar for i in range(1, n_bins)]
    counts = [0] * n_bins
    for v in bersih:
        counts[_bin_index(v, edges)] += 1

    total = len(bersih)
    return {
        "constant": None,
        "edges": edges,
        "proportions": [c / total for c in counts],
        "n": total,
    }


def _bin_index(value, edges):
    for i, batas in enumerate(edges):
        if value < batas:
            return i
    return len(edges)


def psi(baseline, values):
    """Hitung PSI nilai sekarang terhadap baseline.

    Mengembalikan ``None`` bila tidak layak dihitung — baseline tidak ada,
    fiturnya konstan, atau sampelnya terlalu sedikit. ``None`` di sini berarti
    "tidak tahu", dan harus diperlakukan berbeda dari "stabil".
    """
    if not baseline or baseline.get("proportions") is None:
        return None

    bersih = [v for v in values if v is not None]
    if len(bersih) < MIN_SAMPLES:
        return None

    edges = baseline["edges"]
    counts = [0] * (len(edges) + 1)
    for v in bersih:
        counts[_bin_index(v, edges)] += 1

    total = len(bersih)
    nilai = 0.0
    for c, p_latih in zip(counts, baseline["proportions"]):
        p_sekarang = c / total
        # Bin kosong menghasilkan pembagian nol dan logaritma tak hingga.
        # Diberi lantai kecil agar PSI tetap terdefinisi; ini praktik lazim.
        a = max(p_sekarang, _EPS)
        b = max(p_latih, _EPS)
        nilai += (a - b) * math.log(a / b)
    return nilai


def klasifikasi(nilai_psi):
    if nilai_psi is None:
        return "tidak diketahui"
    if nilai_psi < PSI_STABIL:
        return "stabil"
    if nilai_psi < PSI_SIGNIFIKAN:
        return "bergeser"
    return "signifikan"


def check_drift(baselines, samples):
    """Bandingkan sampel terkini terhadap baseline, per fitur.

    ``baselines``: dict ``{nama_fitur: baseline}`` dari ``model_meta.json``.
    ``samples``: dict ``{nama_fitur: [nilai, ...]}`` dari prediksi terkini.
    """
    per_fitur = {}
    for nama, baseline in sorted(baselines.items()):
        nilai = psi(baseline, samples.get(nama, []))
        per_fitur[nama] = {"psi": nilai, "status": klasifikasi(nilai)}

    terukur = [v["psi"] for v in per_fitur.values() if v["psi"] is not None]
    bergeser = [n for n, v in per_fitur.items() if v["status"] == "signifikan"]

    return {
        "per_feature": per_fitur,
        "max_psi": max(terukur) if terukur else None,
        "n_measured": len(terukur),
        "n_unknown": len(per_fitur) - len(terukur),
        "drifted_features": sorted(bergeser),
        "status": "signifikan" if bergeser else (
            "stabil" if terukur else "tidak diketahui"
        ),
        "recommendation": (
            f"Latih ulang model: {len(bergeser)} fitur bergeser signifikan "
            f"({', '.join(sorted(bergeser)[:3])}...)"
            if bergeser else
            "Tidak ada tindakan yang diperlukan"
            if terukur else
            f"Belum cukup sampel (minimal {MIN_SAMPLES} per fitur)"
        ),
    }
