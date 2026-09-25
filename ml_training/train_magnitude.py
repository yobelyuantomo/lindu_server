"""Estimasi dini puncak guncangan dari detik-detik pertama sebuah kejadian.

--------------------------------------------------------------------------
Kenapa BUKAN "regresi magnitudo" seperti rencana awal
--------------------------------------------------------------------------
Rencana semula: latih model untuk memprediksi magnitudo, lalu bandingkan
dengan formula lama ``5.0 + 1.5*log10(pga_max / 0.1)``.

Itu tidak bisa dikerjakan secara jujur. Satu-satunya label magnitudo yang
dimiliki proyek ini ada di ``tb_system_alerts.magnitude`` — dan nilai itu
**dihasilkan oleh formula lama itu sendiri**. Melatih model untuk
memprediksinya berarti melatih model meniru formula. Secara konstruksi model
tidak akan pernah mengalahkannya; paling bagus ia menyamai, dan angka
"perbandingan" yang dihasilkan tidak bermakna apa pun. Katalog magnitudo
independen (mis. BMKG) juga tidak tersedia karena tidak ada gempa tektonik
nyata dalam masa proyek.

Yang dikerjakan sebagai gantinya adalah tugas yang **targetnya terukur dari
data itu sendiri**, sekaligus lebih dekat ke inti sebuah sistem peringatan
dini:

    Dari 1 detik pertama sebuah kejadian, seberapa kuat guncangan ini
    akan menjadi?

Target: PGA puncak sepanjang kejadian. Fitur: hanya dari jendela awal.
Nilainya nyata — memutuskan mengunci katup gas jauh lebih berguna dilakukan
sebelum puncak guncangan tiba, bukan sesudahnya.

Pembandingnya dua, keduanya jujur:

- **Persistensi**: anggap puncak = PGA tertinggi yang sudah terlihat di
  jendela awal. Ini yang secara efektif dilakukan sistem lama.
- **Formula lama** diterapkan pada PGA jendela awal.

Pemakaian::

    python -m ml_training.train_magnitude --labeled data/raw/labeled.csv
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import (  # noqa: E402
    FEATURE_NAMES,
    build_windows,
    extract_features,
    is_analyzable_window,
)
from ml_training.label_dataset import parse_row  # noqa: E402

DEFAULT_SEED = 42
MODEL_FILENAME = "magnitude.joblib"
META_FILENAME = "magnitude_meta.json"

#: Panjang jendela awal yang boleh dilihat model, dalam detik.
LEAD_WINDOW_S = 1.0


def formula_magnitudo(pga):
    """Formula magnitudo Assignment 2 (consensus.py)."""
    aman = max(0.001, pga or 0.0)
    m = 5.0 + 1.5 * math.log10(aman / 0.1)
    return max(3.0, min(9.5, m))


def build_events(rows, lead_window_s=LEAD_WINDOW_S):
    """Susun contoh latih per kejadian.

    Satu kejadian = satu ``session_id`` pada satu node. Fitur diambil hanya
    dari ``lead_window_s`` detik pertama; targetnya PGA puncak sepanjang
    kejadian, termasuk bagian yang belum dilihat model.
    """
    per_sesi = defaultdict(list)
    for row in rows:
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

        puncak = max((r.get("pga") or 0.0) for r in baris)
        puncak_awal = max((r.get("pga") or 0.0) for r in awal)

        events.append({
            "node_id": node_id,
            "session_id": session_id,
            "label": baris[0].get("label"),
            "features": extract_features(awal),
            "peak_pga": puncak,
            "early_peak_pga": puncak_awal,
            "n_rows": len(baris),
        })
    return events


def split_events(events, seed=DEFAULT_SEED, test_ratio=0.3):
    """Bagi per ``session_id`` agar satu kejadian tidak muncul di dua sisi."""
    import random

    sessions = sorted({e["session_id"] for e in events})
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_test = max(1, round(len(sessions) * test_ratio)) if len(sessions) > 1 else 0
    uji = set(sessions[:n_test])
    return (
        [e for e in events if e["session_id"] not in uji],
        [e for e in events if e["session_id"] in uji],
    )


def mae(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    if not y_true:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def evaluate_baselines(events):
    """Metrik dua pembanding, dalam satuan PGA maupun magnitudo."""
    y = [e["peak_pga"] for e in events]
    persistensi = [e["early_peak_pga"] for e in events]

    y_mag = [formula_magnitudo(v) for v in y]
    persistensi_mag = [formula_magnitudo(v) for v in persistensi]

    return {
        "persistensi": {
            "pga_mae": mae(y, persistensi),
            "pga_rmse": rmse(y, persistensi),
            "magnitudo_mae": mae(y_mag, persistensi_mag),
            "magnitudo_rmse": rmse(y_mag, persistensi_mag),
            "keterangan": "anggap puncak = PGA tertinggi yang sudah terlihat di jendela awal",
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", required=True, help="CSV hasil label_dataset.py")
    parser.add_argument("--models-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "models"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lead-window", type=float, default=LEAD_WINDOW_S)
    args = parser.parse_args(argv)

    with open(args.labeled, newline="", encoding="utf-8") as handle:
        rows = [parse_row(r) for r in csv.DictReader(handle)]

    events = build_events(rows, args.lead_window)
    print(f"[OK] {len(events)} kejadian terbentuk dari {len(rows)} baris")
    if len(events) < 6:
        print("[!] Terlalu sedikit kejadian untuk melatih regresi yang bermakna.")
        print("    Tambah sesi perekaman — lihat ml_training/RECORDING_PROTOCOL.md")
        return 1

    latih, uji = split_events(events, args.seed)
    print(f"[i] latih: {len(latih)} kejadian | uji: {len(uji)} kejadian")
    if not uji:
        print("[!] Split uji kosong.")
        return 1

    dasar = evaluate_baselines(uji)
    print("\n=== Pembanding: persistensi ===")
    for k, v in dasar["persistensi"].items():
        if isinstance(v, float):
            print(f"  {k:<16}{v:.4f}")

    import joblib
    from sklearn.ensemble import GradientBoostingRegressor

    X_latih = [[e["features"][n] for n in FEATURE_NAMES] for e in latih]
    y_latih = [e["peak_pga"] for e in latih]
    X_uji = [[e["features"][n] for n in FEATURE_NAMES] for e in uji]
    y_uji = [e["peak_pga"] for e in uji]

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=args.seed
    )
    model.fit(X_latih, y_latih)
    prediksi = list(model.predict(X_uji))

    hasil = {
        "pga_mae": mae(y_uji, prediksi),
        "pga_rmse": rmse(y_uji, prediksi),
        "magnitudo_mae": mae(
            [formula_magnitudo(v) for v in y_uji],
            [formula_magnitudo(v) for v in prediksi],
        ),
    }
    print("\n=== Model (GradientBoostingRegressor) ===")
    for k, v in hasil.items():
        print(f"  {k:<16}{v:.4f}")

    perbaikan = dasar["persistensi"]["pga_mae"] - hasil["pga_mae"]
    print(f"\n  Perbaikan MAE terhadap persistensi: {perbaikan:+.4f} G")
    if perbaikan <= 0:
        print("  [!] Model TIDAK mengalahkan persistensi. Jangan dipakai, dan")
        print("      laporkan apa adanya — ini hasil yang sah dan informatif.")

    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.models_dir, MODEL_FILENAME))
    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "task": "estimasi dini PGA puncak dari jendela awal",
        "target": "peak_pga sepanjang kejadian",
        "lead_window_s": args.lead_window,
        "feature_names": list(FEATURE_NAMES),
        "seed": args.seed,
        "n_train_events": len(latih),
        "n_test_events": len(uji),
        "model_report": hasil,
        "baselines": dasar,
        "beats_persistence": perbaikan > 0,
    }
    with open(os.path.join(args.models_dir, META_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(f"\n[OK] Model + metadata -> {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
