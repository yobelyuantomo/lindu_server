"""Evaluasi sistem rule-based Assignment 2 sebagai pembanding model ML.

Ini bukti utama untuk butir (h) requirement soal: "bandingkan sistem cerdas
dengan sistem rule-based". Agar perbandingannya sah, dua hal dijaga:

1. **Aturan yang dievaluasi adalah aturan yang sesungguhnya.** Ambang batasnya
   diambil dari ``ml.feature_extractor.rule_based_verdict`` — fungsi yang sama
   yang dipakai produksi — bukan disalin ulang di sini. Menyalin ulang membuka
   peluang baseline yang dievaluasi berbeda dari baseline yang berjalan.

2. **Dievaluasi pada split uji yang sama persis** dengan model ML, memakai
   dataset dan pembagian yang identik.

Pemakaian::

    python -m ml_training.evaluate_baseline \\
        --dataset data/processed/dataset_v1.csv --out reports/baseline.json

--------------------------------------------------------------------------
Catatan metodologi
--------------------------------------------------------------------------
Sistem rule-based bekerja per pesan telemetri, sedangkan model ML bekerja per
jendela. Agar keduanya dinilai pada satuan yang sama, verdict rule-based di
tingkat jendela didefinisikan sebagai: **jendela dianggap gempa bila ada
minimal satu sampel di dalamnya yang lolos filter tiga lapis.**

Definisi itu dipilih karena memang begitulah sistem lama berperilaku — satu
pesan yang lolos filter sudah cukup untuk masuk trigger buffer dan memicu
proses konsensus. Memakai definisi lain (misalnya mayoritas sampel) akan
membuat baseline terlihat lebih baik daripada sistem yang benar-benar berjalan,
dan itu menguntungkan ML secara tidak jujur.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import (  # noqa: E402
    RULE_FREQ_HZ_MAX,
    RULE_PGA_MIN,
    RULE_STA_LTA_MIN,
)
from ml_training.metrics import classification_report, format_report  # noqa: E402

POSITIVE_LABEL = "earthquake"
NEGATIVE_LABEL = "noise"


def predict_window(features):
    """Verdict rule-based untuk satu jendela, dari vektor fiturnya.

    Fitur jendela sudah memuat nilai ekstrem tiap metrik, sehingga "ada minimal
    satu sampel yang lolos ketiga lapis" dapat dinilai langsung:

    - ``pga_max``     : sampel terkuat di jendela
    - ``sta_lta_max`` : rasio energi tertinggi
    - ``freq_min``    : frekuensi terendah (paling mungkin lolos ``<= 20 Hz``)

    Ini adalah pendekatan optimistis terhadap baseline — kalau ada kombinasi
    yang bisa lolos, dianggap lolos. Disengaja, supaya baseline tidak
    dirugikan oleh cara pengukuran.
    """
    pga_max = features.get("pga_max")
    sta_max = features.get("sta_lta_max")
    freq_min = features.get("freq_min")
    if pga_max is None or sta_max is None or freq_min is None:
        return NEGATIVE_LABEL
    lolos = (
        pga_max >= RULE_PGA_MIN
        and sta_max >= RULE_STA_LTA_MIN
        and freq_min <= RULE_FREQ_HZ_MAX
    )
    return POSITIVE_LABEL if lolos else NEGATIVE_LABEL


def load_dataset(path, split=None):
    """Baca dataset fitur. ``split`` menyaring kolom ``split`` bila diisi."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if split and raw.get("split") != split:
                continue
            row = dict(raw)
            for key, value in raw.items():
                if key in ("split", "label", "session_id", "scenario", "node_id"):
                    continue
                if key == "is_augmented":
                    row[key] = str(value).lower() == "true"
                    continue
                try:
                    row[key] = float(value) if value not in (None, "") else None
                except (TypeError, ValueError):
                    row[key] = None
            rows.append(row)
    return rows


def evaluate(rows):
    """Jalankan baseline pada baris dataset, kembalikan laporan metrik."""
    y_true = [r["label"] for r in rows]
    y_pred = [predict_window(r) for r in rows]
    return classification_report(y_true, y_pred, POSITIVE_LABEL), y_pred


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="CSV hasil build_dataset.py")
    parser.add_argument("--split", default="test", help="split yang dievaluasi (default: test)")
    parser.add_argument("--out", help="Path JSON laporan")
    args = parser.parse_args(argv)

    rows = load_dataset(args.dataset, args.split)
    if not rows:
        print(f"[!] Tidak ada baris untuk split '{args.split}' di {args.dataset}")
        return 1

    # Baris augmentasi dibuang: ia hanya ada di train, dan mengevaluasi baseline
    # pada data sintetis tidak mencerminkan performa sesungguhnya.
    asli = [r for r in rows if not r.get("is_augmented")]
    if len(asli) != len(rows):
        print(f"[i] {len(rows) - len(asli)} baris augmentasi diabaikan")

    report, _ = evaluate(asli)
    report["split"] = args.split
    report["rule"] = {
        "pga_min": RULE_PGA_MIN,
        "sta_lta_min": RULE_STA_LTA_MIN,
        "freq_hz_max": RULE_FREQ_HZ_MAX,
        "window_verdict": "ada minimal satu sampel yang lolos ketiga lapis",
    }

    print(format_report(report, f"Baseline rule-based (split: {args.split})"))

    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\n[OK] Laporan ditulis ke {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
