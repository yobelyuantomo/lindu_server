"""Latih classifier kejadian seismik (earthquake / noise / gas_leak).

Pemakaian::

    python -m ml_training.train_classifier \\
        --dataset data/processed/dataset_v1.csv \\
        --models-dir ml/models

Menghasilkan ``ml/models/classifier.joblib`` dan ``ml/models/model_meta.json``.

--------------------------------------------------------------------------
Aturan yang dipegang
--------------------------------------------------------------------------
**Pemilihan model memakai split val, bukan test.** Test hanya disentuh sekali
di akhir untuk angka yang dilaporkan. Memilih model berdasarkan skor test
berarti test sudah ikut melatih, dan angkanya berhenti menjadi estimasi
performa pada data baru.

**Class weighting, bukan oversampling.** Kelas earthquake hampir pasti lebih
sedikit daripada noise. Oversampling pada data yang sudah diaugmentasi
berisiko menggandakan varian dari jendela yang sama berkali-kali; class
weighting mencapai tujuan yang sama tanpa menduplikasi apa pun.

**Baris augmentasi hanya dipakai saat fit.** Validasi dan uji selalu memakai
jendela asli, supaya metriknya mencerminkan data sungguhan.

**Urutan fitur disimpan di metadata.** Inference engine menolak memuat model
bila urutannya tidak cocok dengan feature_extractor — ini yang mencegah
train/serve skew berubah menjadi bug senyap.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402
from ml_training.evaluate_baseline import load_dataset  # noqa: E402
from ml_training.metrics import (  # noqa: E402
    classification_report,
    format_report,
    label_distribution,
)

DEFAULT_SEED = 42
MODEL_FILENAME = "classifier.joblib"
META_FILENAME = "model_meta.json"


def split_xy(rows, include_augmented=True):
    """Pisahkan matriks fitur dan label dari baris dataset."""
    dipakai = rows if include_augmented else [r for r in rows if not r.get("is_augmented")]
    X = [[r.get(name) or 0.0 for name in FEATURE_NAMES] for r in dipakai]
    y = [r["label"] for r in dipakai]
    return X, y, dipakai


def build_candidates(seed=DEFAULT_SEED):
    """Kandidat model yang diadu pada split val.

    Random Forest dan Gradient Boosting dipilih sebagai baseline karena bekerja
    baik pada fitur hasil rekayasa berjumlah puluhan dan dataset berukuran
    ratusan sampai ribuan baris — kondisi yang realistis untuk proyek ini.
    Jaringan saraf tidak dipakai di tahap ini: dengan data sesedikit itu ia
    hampir pasti kalah, sekaligus jauh lebih sulit dijelaskan saat sidang.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
    }


def train_and_select(train_rows, val_rows, seed=DEFAULT_SEED):
    """Latih tiap kandidat, pilih yang macro F1-nya terbaik pada val."""
    X_train, y_train, _ = split_xy(train_rows, include_augmented=True)
    X_val, y_val, _ = split_xy(val_rows, include_augmented=False)

    hasil = {}
    terbaik, model_terbaik, skor_terbaik = None, None, -1.0
    for nama, model in build_candidates(seed).items():
        model.fit(X_train, y_train)
        laporan = classification_report(y_val, list(model.predict(X_val)))
        hasil[nama] = laporan
        print(format_report(laporan, f"{nama} (val)"))
        if laporan["macro_f1"] > skor_terbaik:
            terbaik, skor_terbaik = nama, laporan["macro_f1"]
            model_terbaik = model

    return model_terbaik, terbaik, hasil


def build_feature_baselines(rows):
    """Histogram distribusi tiap fitur pada data latih.

    Dipakai ``ml/drift_monitor.py`` untuk mendeteksi pergeseran kondisi
    lapangan. Baris augmentasi ikut disertakan karena baseline harus
    mencerminkan distribusi yang benar-benar dilihat model saat fit.
    """
    from ml.drift_monitor import build_baseline

    return {
        nama: build_baseline([r.get(nama) for r in rows])
        for nama in FEATURE_NAMES
    }


def feature_importance(model):
    """Kepentingan fitur, bila model mendukungnya."""
    nilai = getattr(model, "feature_importances_", None)
    if nilai is None:
        return {}
    pasangan = sorted(zip(FEATURE_NAMES, (float(v) for v in nilai)), key=lambda p: -p[1])
    return dict(pasangan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "models"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--version", default=None, help="versi model (default: cap waktu)")
    args = parser.parse_args(argv)

    import joblib

    train_rows = load_dataset(args.dataset, "train")
    val_rows = load_dataset(args.dataset, "val")
    test_rows = load_dataset(args.dataset, "test")

    for nama, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        asli = [r for r in rows if not r.get("is_augmented")]
        print(f"[i] {nama:<6} {len(rows):>5} baris ({len(asli)} asli) "
              f"{label_distribution([r['label'] for r in asli])}")

    if not train_rows or not val_rows or not test_rows:
        print("\n[!] Ada split yang kosong. Dataset belum cukup untuk melatih model.")
        print("    Tambah sesi perekaman — lihat ml_training/RECORDING_PROTOCOL.md")
        return 1

    kelas_train = set(r["label"] for r in train_rows)
    if len(kelas_train) < 2:
        print(f"\n[!] Split train hanya punya kelas {kelas_train}. Butuh minimal dua.")
        return 1

    model, nama_terbaik, hasil_val = train_and_select(train_rows, val_rows, args.seed)
    print(f"\n[OK] Model terpilih berdasarkan macro F1 pada val: {nama_terbaik}")

    # Test disentuh SEKALI, setelah model dipilih.
    X_test, y_test, _ = split_xy(test_rows, include_augmented=False)
    laporan_test = classification_report(y_test, list(model.predict(X_test)))
    print(format_report(laporan_test, f"{nama_terbaik} (TEST - angka yang dilaporkan)"))

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, MODEL_FILENAME)
    joblib.dump(model, model_path)

    versi = args.version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    meta = {
        "model_version": versi,
        "model_type": nama_terbaik,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "feature_names": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "classes": sorted(set(y_test) | kelas_train),
        "dataset": os.path.basename(args.dataset),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_test_rows": len(test_rows),
        "val_reports": hasil_val,
        "test_report": laporan_test,
        "feature_importance": feature_importance(model),
        "selection_metric": "macro_f1 pada split val",
        # Histogram distribusi fitur saat pelatihan. Disimpan di sini supaya
        # runtime bisa mendeteksi drift tanpa perlu memuat ulang dataset latih.
        "feature_baselines": build_feature_baselines(train_rows),
    }
    meta_path = os.path.join(args.models_dir, META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"\n[OK] Model  -> {model_path}")
    print(f"[OK] Metadata -> {meta_path}")
    print(f"[OK] Versi model: {versi}")

    penting = list(meta["feature_importance"].items())[:5]
    if penting:
        print("\n  Lima fitur paling berpengaruh:")
        for nama, nilai in penting:
            print(f"      {nama:<24}{nilai:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
