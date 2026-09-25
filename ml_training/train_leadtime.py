"""Prediksi sisa waktu sampai puncak guncangan (lead-time).

Inti sebuah sistem peringatan dini bukan sekadar mengetahui bahwa gempa sedang
terjadi, melainkan **berapa detik tersisa** sebelum guncangan terkuat tiba.
Itulah yang menentukan apakah orang sempat menjauh dari jendela dan apakah
katup gas sempat ditutup.

Target diberi label secara retrospektif dari rekaman: selisih waktu antara
akhir jendela awal yang dilihat model dan saat PGA puncak benar-benar terjadi.
Tidak ada label dari luar yang dibutuhkan — semuanya terukur dari data sendiri.

Pemakaian::

    python -m ml_training.train_leadtime --labeled data/raw/labeled.csv

--------------------------------------------------------------------------
Batasan yang harus dinyatakan di laporan
--------------------------------------------------------------------------
Model ini diposisikan sebagai **proof-of-concept**, bukan kapabilitas siap
pakai. Dua alasan:

1. Kelas ``earthquake`` berasal dari peragaan meja getar, bukan gempa
   tektonik. Pada gempa sungguhan, jarak antara gelombang-P dan gelombang-S
   ditentukan oleh jarak episentrum — hubungan fisika yang sama sekali tidak
   terwakili dalam goyangan tangan di atas meja.

2. Jumlah kejadian sangat sedikit, sehingga MAE yang dilaporkan punya
   ketidakpastian besar.

Nilainya tetap ada: jalur datanya terbukti berfungsi dan siap dipakai begitu
rekaman gempa sungguhan tersedia.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    is_analyzable_window,
)
from ml_training.label_dataset import parse_row  # noqa: E402
from ml_training.train_magnitude import mae, rmse, split_events  # noqa: E402

DEFAULT_SEED = 42
MODEL_FILENAME = "leadtime.joblib"
META_FILENAME = "leadtime_meta.json"

LEAD_WINDOW_S = 1.0
TARGET_LABEL = "earthquake"


def build_events(rows, lead_window_s=LEAD_WINDOW_S):
    """Susun contoh latih: fitur jendela awal -> sisa detik menuju puncak.

    Hanya kejadian ``earthquake`` yang dipakai. Melatih lead-time pada getaran
    non-seismik tidak bermakna: tidak ada "puncak gelombang-S" yang sedang
    menuju ke mana pun.
    """
    per_sesi = defaultdict(list)
    for row in rows:
        if row.get("label") != TARGET_LABEL:
            continue
        kunci = (row.get("node_id"), row.get("session_id"))
        if kunci[0] and kunci[1]:
            per_sesi[kunci].append(row)

    events = []
    for (node_id, session_id), baris in sorted(per_sesi.items()):
        baris.sort(key=lambda r: (r.get("sensor_ts") or 0.0))
        t0 = baris[0].get("sensor_ts")
        if t0 is None:
            continue

        awal = [r for r in baris if (r.get("sensor_ts") or 0.0) - t0 < lead_window_s]
        if not is_analyzable_window(awal):
            continue

        # Waktu saat PGA puncak benar-benar terjadi.
        puncak = max(baris, key=lambda r: (r.get("pga") or 0.0))
        t_puncak = (puncak.get("sensor_ts") or t0) - t0

        # Lead-time diukur dari AKHIR jendela yang dilihat model, bukan dari
        # awalnya. Model baru bisa memberi keputusan setelah jendela lengkap,
        # jadi mengukur dari awal jendela akan melebih-lebihkan waktu yang
        # sebenarnya tersedia.
        t_akhir_jendela = (awal[-1].get("sensor_ts") or t0) - t0
        lead = t_puncak - t_akhir_jendela

        # Puncak yang sudah lewat sebelum model sempat memutuskan tidak bisa
        # diperingatkan; dibuang alih-alih dilatihkan sebagai lead-time negatif.
        if lead <= 0:
            continue

        events.append({
            "node_id": node_id,
            "session_id": session_id,
            "features": extract_features(awal),
            "lead_time": lead,
            "peak_pga": puncak.get("pga") or 0.0,
        })
    return events


def evaluate_baseline(events):
    """Pembanding: selalu tebak lead-time rata-rata data latih.

    Pembanding yang tampak sepele, tetapi justru itu gunanya. Model regresi
    yang tidak bisa mengalahkan tebakan konstan berarti tidak mempelajari
    apa pun dari sinyalnya.
    """
    if not events:
        return None
    y = [e["lead_time"] for e in events]
    rata = sum(y) / len(y)
    tebakan = [rata] * len(y)
    return {
        "konstanta": rata,
        "mae": mae(y, tebakan),
        "rmse": rmse(y, tebakan),
        "keterangan": "selalu menebak lead-time rata-rata",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", required=True)
    parser.add_argument("--models-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "models"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lead-window", type=float, default=LEAD_WINDOW_S)
    args = parser.parse_args(argv)

    with open(args.labeled, newline="", encoding="utf-8") as handle:
        rows = [parse_row(r) for r in csv.DictReader(handle)]

    events = build_events(rows, args.lead_window)
    print(f"[OK] {len(events)} kejadian '{TARGET_LABEL}' dengan lead-time positif")
    if len(events) < 6:
        print("[!] Terlalu sedikit kejadian untuk melatih regresi yang bermakna.")
        print("    Tambah sesi perekaman — lihat ml_training/RECORDING_PROTOCOL.md")
        return 1

    latih, uji = split_events(events, args.seed)
    print(f"[i] latih: {len(latih)} | uji: {len(uji)}")
    if not uji:
        print("[!] Split uji kosong.")
        return 1

    dasar = evaluate_baseline(uji)
    print(f"\n=== Pembanding: tebakan konstan ({dasar['konstanta']:.2f} s) ===")
    print(f"  MAE  {dasar['mae']:.4f} s")
    print(f"  RMSE {dasar['rmse']:.4f} s")

    import joblib
    from sklearn.ensemble import GradientBoostingRegressor

    X_latih = [[e["features"][n] for n in FEATURE_NAMES] for e in latih]
    y_latih = [e["lead_time"] for e in latih]
    X_uji = [[e["features"][n] for n in FEATURE_NAMES] for e in uji]
    y_uji = [e["lead_time"] for e in uji]

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=args.seed
    )
    model.fit(X_latih, y_latih)
    prediksi = [float(v) for v in model.predict(X_uji)]

    hasil = {"mae": float(mae(y_uji, prediksi)), "rmse": float(rmse(y_uji, prediksi))}
    print("\n=== Model (GradientBoostingRegressor) ===")
    print(f"  MAE  {hasil['mae']:.4f} s")
    print(f"  RMSE {hasil['rmse']:.4f} s")

    perbaikan = dasar["mae"] - hasil["mae"]
    print(f"\n  Perbaikan MAE terhadap tebakan konstan: {perbaikan:+.4f} s")
    if perbaikan <= 0:
        print("  [!] Model TIDAK mengalahkan tebakan konstan — artinya ia belum")
        print("      mempelajari apa pun dari sinyalnya. Laporkan apa adanya.")

    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.models_dir, MODEL_FILENAME))
    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "task": "prediksi sisa detik menuju puncak guncangan",
        "target": "selisih waktu antara akhir jendela awal dan PGA puncak",
        "lead_window_s": args.lead_window,
        "feature_names": list(FEATURE_NAMES),
        "seed": args.seed,
        "n_train_events": len(latih),
        "n_test_events": len(uji),
        "model_report": hasil,
        "baseline": dasar,
        "beats_constant": bool(perbaikan > 0),
        "status": "proof-of-concept",
        "limitations": [
            "kelas earthquake berasal dari peragaan meja getar, bukan gempa tektonik",
            "hubungan jarak episentrum terhadap selisih P-S tidak terwakili",
            "jumlah kejadian sedikit sehingga MAE punya ketidakpastian besar",
        ],
    }
    with open(os.path.join(args.models_dir, META_FILENAME), "w", encoding="utf-8") as h:
        json.dump(meta, h, indent=2, ensure_ascii=False)
    print(f"\n[OK] Model + metadata -> {args.models_dir}")
    print("[i] Status: proof-of-concept. Batasannya tercatat di metadata dan")
    print("    wajib ikut ditulis di bab Limitasi laporan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
