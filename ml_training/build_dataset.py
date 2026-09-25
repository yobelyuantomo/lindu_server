"""Bangun dataset fitur siap latih dari telemetri berlabel.

Alur: baris berlabel -> jendela -> augmentasi (khusus train) -> vektor fitur.

Pemakaian::

    python -m ml_training.build_dataset \\
        --labeled data/raw/labeled.csv --out data/processed/dataset_v1.csv

Dua hal yang menjaga agar metriknya tidak palsu:

**Split per kejadian, bukan per baris.** Jendela dari satu sesi perekaman yang
sama sangat mirip satu sama lain. Kalau dipecah acak per baris, potongan sesi
yang sama muncul di train dan test sekaligus, dan model tinggal menghafal.
Akurasinya akan terlihat hampir sempurna dan sepenuhnya menyesatkan. Karena
itu pemecahan dilakukan di tingkat ``session_id``.

**Augmentasi hanya pada train.** Menambah salinan data ke validation atau test
berarti menguji model pada turunan dari contoh yang sudah pernah dilihatnya.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import (  # noqa: E402
    FEATURE_NAMES,
    WINDOW_SECONDS,
    build_windows,
    extract_features,
    is_analyzable_window,
)
from ml_training.label_dataset import parse_row  # noqa: E402

DEFAULT_SEED = 42
DEFAULT_SPLIT = (0.70, 0.15, 0.15)

#: Kolom metadata yang ikut ditulis di samping fitur.
META_COLUMNS = ["split", "label", "session_id", "scenario", "node_id", "is_augmented"]


def windows_from_rows(rows, window_seconds=WINDOW_SECONDS):
    """Ubah baris berlabel menjadi daftar jendela yang layak dianalisis.

    Jendela hanya diterima bila seluruh barisnya berasal dari satu sesi yang
    sama; jendela yang menyeberangi batas sesi dibuang karena labelnya ambigu.
    """
    per_node = defaultdict(list)
    for row in rows:
        per_node[row["node_id"]].append(row)

    hasil = []
    for node_id, node_rows in per_node.items():
        node_rows.sort(key=lambda r: (r.get("sensor_ts") or 0.0))
        for window in build_windows(node_rows, window_seconds):
            sessions = {r.get("session_id") for r in window}
            if len(sessions) != 1:
                continue
            if not is_analyzable_window(window):
                continue
            first = window[0]
            hasil.append(
                {
                    "rows": window,
                    "label": first["label"],
                    "session_id": first["session_id"],
                    "scenario": first.get("scenario", ""),
                    "node_id": node_id,
                }
            )
    return hasil


def split_sessions(windows, split=DEFAULT_SPLIT, seed=DEFAULT_SEED):
    """Bagi **sesi** (bukan jendela) ke train/val/test, distratifikasi per label.

    Stratifikasi dilakukan dengan memecah daftar sesi tiap label secara
    terpisah, supaya kelas minoritas tidak habis di satu split saja.
    """
    label_of_session = {}
    for w in windows:
        label_of_session[w["session_id"]] = w["label"]

    per_label = defaultdict(list)
    for session_id, label in label_of_session.items():
        per_label[label].append(session_id)

    rng = random.Random(seed)
    assignment = {}
    for label, sessions in sorted(per_label.items()):
        sessions = sorted(sessions)
        rng.shuffle(sessions)
        n_val, n_test = _split_sizes(len(sessions), split)
        n_train = len(sessions) - n_val - n_test
        for i, session_id in enumerate(sessions):
            if i < n_train:
                assignment[session_id] = "train"
            elif i < n_train + n_val:
                assignment[session_id] = "val"
            else:
                assignment[session_id] = "test"
    return assignment


def _split_sizes(n, split):
    """Tentukan jumlah sesi untuk val dan test.

    Pembulatan proporsional saja tidak cukup: dengan 5 sesi per kelas,
    ``round(5 * 0.15)`` menghasilkan 1 untuk val dan menyisakan 0 untuk test —
    artinya tidak ada data held-out sama sekali, dan metrik uji menjadi tidak
    ada artinya. Karena itu val dan test dijamin kebagian minimal satu sesi
    selama jumlahnya memungkinkan.
    """
    if n <= 1:
        return 0, 0
    if n == 2:
        return 0, 1  # satu untuk latih, satu untuk uji
    return max(1, round(n * split[1])), max(1, round(n * split[2]))


def augment_window(rows, rng):
    """Hasilkan satu varian jendela: skala amplitudo + derau kecil.

    Pergeseran waktu tidak diterapkan karena fitur di modul ini seluruhnya
    relatif terhadap awal jendela, sehingga menggesernya tidak mengubah apa pun.
    """
    skala = rng.uniform(0.8, 1.25)
    hasil = []
    for row in rows:
        baru = dict(row)
        for kolom in ("pga", "accel_x", "accel_y", "accel_z"):
            nilai = row.get(kolom)
            if nilai is not None:
                derau = rng.gauss(0.0, abs(nilai) * 0.03)
                baru[kolom] = nilai * skala + derau
        sta = row.get("sta_lta")
        if sta is not None:
            baru["sta_lta"] = max(0.0, sta + rng.gauss(0.0, 0.05))
        hasil.append(baru)
    return hasil


def build_rows(windows, assignment, augment_factor=2, seed=DEFAULT_SEED):
    """Ubah jendela menjadi baris dataset (fitur + metadata)."""
    rng = random.Random(seed)
    dataset = []
    for w in windows:
        split = assignment.get(w["session_id"], "train")
        dataset.append(
            {
                **extract_features(w["rows"]),
                "split": split,
                "label": w["label"],
                "session_id": w["session_id"],
                "scenario": w["scenario"],
                "node_id": w["node_id"],
                "is_augmented": False,
            }
        )
        if split != "train":
            continue
        for _ in range(augment_factor):
            dataset.append(
                {
                    **extract_features(augment_window(w["rows"], rng)),
                    "split": split,
                    "label": w["label"],
                    "session_id": w["session_id"],
                    "scenario": w["scenario"],
                    "node_id": w["node_id"],
                    "is_augmented": True,
                }
            )
    return dataset


def write_dataset(dataset, path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = META_COLUMNS + FEATURE_NAMES
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return len(dataset)


def build_meta(dataset, assignment, seed, split, window_seconds, augment_factor):
    per_split = defaultdict(Counter)
    for row in dataset:
        per_split[row["split"]][row["label"]] += 1
    return {
        "seed": seed,
        "split_ratio": list(split),
        "window_seconds": window_seconds,
        "augment_factor": augment_factor,
        "split_unit": "session_id",
        "feature_names": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "n_rows": len(dataset),
        "n_sessions": len(set(assignment)),
        "sessions_per_split": dict(Counter(assignment.values())),
        "rows_per_split": {k: dict(v) for k, v in per_split.items()},
        "n_augmented": sum(1 for r in dataset if r["is_augmented"]),
    }


def _warn_if_thin(meta):
    masalah = []
    for split in ("train", "val", "test"):
        isi = meta["rows_per_split"].get(split, {})
        if not isi:
            masalah.append(f"split '{split}' kosong")
        elif len(isi) < 2:
            masalah.append(f"split '{split}' hanya punya kelas {list(isi)}")
    if meta["n_sessions"] < 6:
        masalah.append(
            f"hanya {meta['n_sessions']} sesi — terlalu sedikit untuk split "
            "per kejadian yang bermakna"
        )
    for m in masalah:
        print(f"[!] {m}")
    if masalah:
        print(
            "    Dataset masih terlalu tipis. Tambah sesi perekaman "
            "(lihat RECORDING_PROTOCOL.md) sebelum menarik kesimpulan dari metrik."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", required=True, help="CSV hasil label_dataset.py")
    parser.add_argument("--out", required=True, help="CSV dataset keluaran")
    parser.add_argument("--meta", help="Path JSON metadata (default: <out>.meta.json)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument(
        "--augment-factor", type=int, default=2,
        help="jumlah salinan tambahan per jendela train (0 = tanpa augmentasi)",
    )
    args = parser.parse_args(argv)

    with open(args.labeled, newline="", encoding="utf-8") as handle:
        rows = [parse_row(r) for r in csv.DictReader(handle)]
    print(f"[OK] {len(rows)} baris berlabel dibaca dari {args.labeled}")

    windows = windows_from_rows(rows, args.window_seconds)
    print(f"[OK] {len(windows)} jendela layak dianalisis")
    if not windows:
        print("[!] Tidak ada jendela yang layak. Dataset tidak bisa dibangun.")
        return 1

    assignment = split_sessions(windows, DEFAULT_SPLIT, args.seed)
    dataset = build_rows(windows, assignment, args.augment_factor, args.seed)
    written = write_dataset(dataset, args.out)

    meta = build_meta(
        dataset, assignment, args.seed, DEFAULT_SPLIT,
        args.window_seconds, args.augment_factor,
    )
    meta_path = args.meta or (os.path.splitext(args.out)[0] + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"[OK] {written} baris dataset ditulis ke {args.out}")
    print(f"[OK] metadata ditulis ke {meta_path}")
    for split in ("train", "val", "test"):
        isi = meta["rows_per_split"].get(split, {})
        total = sum(isi.values())
        rincian = ", ".join(f"{k}={v}" for k, v in sorted(isi.items())) or "-"
        print(f"      {split:<6} {total:>5} baris  ({rincian})")
    _warn_if_thin(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
