"""Deteksi anomali tanpa label, dilatih per node.

Classifier hanya mengenali pola yang pernah dilatihkan. Untuk pola getaran
yang belum pernah ditemui — jenis gangguan baru, sensor yang mulai rusak,
pemasangan yang bergeser — dibutuhkan jalur yang tidak bergantung pada label.

Model dilatih **hanya pada getaran normal** milik node itu sendiri, yaitu
jendela berlabel ``noise``. Baseline dibuat per node karena kebisingan
lingkungan tiap node berbeda: node di dekat jalan raya tidak sebanding dengan
node di ruang tenang, dan memaksakan satu baseline bersama akan membuat node
yang ramai selamanya tampak anomali.

Evaluasinya memakai kelas ``earthquake`` sebagai penguji: model yang tidak
pernah melihat gempa sama sekali seharusnya menilainya sebagai anomali.
ROC-AUC dari pemisahan itulah yang dilaporkan.

Pemakaian::

    python -m ml_training.train_anomaly --dataset data/processed/dataset_v1.csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402
from ml_training.evaluate_baseline import load_dataset  # noqa: E402

DEFAULT_SEED = 42
MODEL_FILENAME = "anomaly.joblib"
META_FILENAME = "anomaly_meta.json"

#: Jumlah jendela normal minimum agar baseline sebuah node layak dibentuk.
MIN_NORMAL_PER_NODE = 20

NORMAL_LABEL = "noise"
NOVEL_LABEL = "earthquake"


def roc_auc(skor_normal, skor_anomali):
    """ROC-AUC lewat statistik Mann-Whitney U, tanpa numpy.

    Nilainya sama dengan peluang bahwa satu sampel anomali yang dipilih acak
    mendapat skor lebih tinggi daripada satu sampel normal yang dipilih acak.
    """
    if not skor_normal or not skor_anomali:
        return None

    semua = [(s, 0) for s in skor_normal] + [(s, 1) for s in skor_anomali]
    semua.sort(key=lambda p: p[0])

    # Ranking rata-rata untuk nilai yang sama, agar seri tidak bias.
    peringkat = [0.0] * len(semua)
    i = 0
    while i < len(semua):
        j = i
        while j + 1 < len(semua) and semua[j + 1][0] == semua[i][0]:
            j += 1
        rata = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            peringkat[k] = rata
        i = j + 1

    jumlah_anomali = sum(p for s, p in semua)
    jumlah_normal = len(semua) - jumlah_anomali
    if not jumlah_anomali or not jumlah_normal:
        return None

    total_peringkat = sum(r for r, (_, p) in zip(peringkat, semua) if p == 1)
    u = total_peringkat - jumlah_anomali * (jumlah_anomali + 1) / 2.0
    return u / (jumlah_anomali * jumlah_normal)


def kelompokkan_per_node(rows):
    per_node = defaultdict(lambda: {"normal": [], "novel": []})
    for row in rows:
        node_id = row.get("node_id")
        if not node_id:
            continue
        if row.get("label") == NORMAL_LABEL:
            per_node[node_id]["normal"].append(row)
        elif row.get("label") == NOVEL_LABEL:
            per_node[node_id]["novel"].append(row)
    return per_node


def _vektor(rows):
    return [[r.get(n) or 0.0 for n in FEATURE_NAMES] for r in rows]


def latih_per_node(per_node, seed=DEFAULT_SEED):
    """Latih satu IsolationForest per node. Kembalikan (model, laporan)."""
    from sklearn.ensemble import IsolationForest

    model_per_node, laporan = {}, {}
    for node_id, data in sorted(per_node.items()):
        normal = data["normal"]
        if len(normal) < MIN_NORMAL_PER_NODE:
            laporan[node_id] = {
                "status": "dilewati",
                "alasan": f"hanya {len(normal)} jendela normal "
                          f"(minimal {MIN_NORMAL_PER_NODE})",
            }
            continue

        model = IsolationForest(
            n_estimators=200, contamination="auto", random_state=seed
        )
        model.fit(_vektor(normal))
        model_per_node[node_id] = model

        # score_samples: makin RENDAH makin anomali. Dibalik tandanya supaya
        # "skor anomali tinggi = makin aneh", yang jauh lebih mudah dibaca di
        # dashboard dan tidak gampang tertukar arah saat menyetel ambang.
        skor_normal = [-float(s) for s in model.score_samples(_vektor(normal))]
        skor_novel = (
            [-float(s) for s in model.score_samples(_vektor(data["novel"]))]
            if data["novel"] else []
        )

        laporan[node_id] = {
            "status": "dilatih",
            "n_normal": len(normal),
            "n_novel": len(data["novel"]),
            "roc_auc": roc_auc(skor_normal, skor_novel),
            "ambang_p95": sorted(skor_normal)[int(len(skor_normal) * 0.95) - 1],
            "skor_normal_rata": sum(skor_normal) / len(skor_normal),
            "skor_novel_rata": (
                sum(skor_novel) / len(skor_novel) if skor_novel else None
            ),
        }
    return model_per_node, laporan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "models"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    import joblib

    # Baris augmentasi dibuang: baseline kebisingan harus mencerminkan apa yang
    # benar-benar terekam di lapangan, bukan turunan sintetisnya.
    rows = [r for r in load_dataset(args.dataset) if not r.get("is_augmented")]
    per_node = kelompokkan_per_node(rows)
    print(f"[OK] {len(rows)} jendela asli dari {len(per_node)} node")

    model_per_node, laporan = latih_per_node(per_node, args.seed)
    if not model_per_node:
        print("\n[!] Tidak ada node yang punya cukup jendela normal.")
        print(f"    Minimal {MIN_NORMAL_PER_NODE} jendela berlabel '{NORMAL_LABEL}' per node.")
        return 1

    print()
    for node_id, r in sorted(laporan.items()):
        if r["status"] != "dilatih":
            print(f"  {node_id:<12} dilewati — {r['alasan']}")
            continue
        auc = r["roc_auc"]
        auc_txt = f"{auc:.3f}" if auc is not None else "—"
        print(
            f"  {node_id:<12} normal={r['n_normal']:<4} novel={r['n_novel']:<4} "
            f"ROC-AUC={auc_txt}"
        )

    auc_terukur = [r["roc_auc"] for r in laporan.values()
                   if r.get("roc_auc") is not None]
    if auc_terukur:
        rata = sum(auc_terukur) / len(auc_terukur)
        print(f"\n  ROC-AUC rata-rata: {rata:.3f}")
        if rata < 0.6:
            print("  [!] Pemisahan lemah. Deteksi anomali ini belum layak dipakai")
            print("      untuk mengambil keputusan; laporkan apa adanya.")

    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(model_per_node, os.path.join(args.models_dir, MODEL_FILENAME))
    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "task": "deteksi anomali per node (unsupervised)",
        "trained_on": f"jendela berlabel '{NORMAL_LABEL}' milik masing-masing node",
        "evaluated_against": f"jendela berlabel '{NOVEL_LABEL}' sebagai penguji",
        "score_convention": "makin tinggi makin anomali (tanda score_samples dibalik)",
        "feature_names": list(FEATURE_NAMES),
        "seed": args.seed,
        "nodes": laporan,
        "mean_roc_auc": (sum(auc_terukur) / len(auc_terukur)) if auc_terukur else None,
    }
    with open(os.path.join(args.models_dir, META_FILENAME), "w", encoding="utf-8") as h:
        json.dump(meta, h, indent=2, ensure_ascii=False)
    print(f"\n[OK] Model + metadata -> {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
