"""Metrik klasifikasi tanpa dependensi eksternal.

Dipakai oleh ``evaluate_baseline.py`` (yang sengaja tidak memakai scikit-learn
agar baseline bisa dievaluasi di mana saja) dan oleh laporan perbandingan.

Definisi di sini mengikuti konvensi scikit-learn supaya angka dari kedua jalur
dapat disandingkan langsung tanpa perlu menjelaskan selisih metodologi.
"""

from collections import Counter


def confusion_matrix(y_true, y_pred, labels=None):
    """Matriks kebingungan sebagai dict ``{(asli, prediksi): jumlah}``."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    matrix = {(a, p): 0 for a in labels for p in labels}
    for a, p in zip(y_true, y_pred):
        matrix[(a, p)] = matrix.get((a, p), 0) + 1
    return matrix, labels


def precision_recall_f1(y_true, y_pred, positive_label):
    """Precision, recall, dan F1 untuk satu kelas (one-vs-rest)."""
    tp = sum(1 for a, p in zip(y_true, y_pred) if a == positive_label and p == positive_label)
    fp = sum(1 for a, p in zip(y_true, y_pred) if a != positive_label and p == positive_label)
    fn = sum(1 for a, p in zip(y_true, y_pred) if a == positive_label and p != positive_label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(1 for a, p in zip(y_true, y_pred) if a == p) / len(y_true)


def false_positive_rate(y_true, y_pred, positive_label):
    """Proporsi kejadian BUKAN positif yang salah diprediksi sebagai positif.

    Ini metrik yang paling relevan untuk Lindu-EEW: sirine yang berbunyi karena
    truk lewat jauh lebih merusak kepercayaan pengguna daripada sekadar angka
    akurasi yang turun sedikit.
    """
    negatif = sum(1 for a in y_true if a != positive_label)
    if not negatif:
        return 0.0
    fp = sum(1 for a, p in zip(y_true, y_pred) if a != positive_label and p == positive_label)
    return fp / negatif


def classification_report(y_true, y_pred, positive_label="earthquake"):
    """Ringkasan metrik lengkap sebagai dict siap diserialisasi."""
    labels = sorted(set(y_true) | set(y_pred))
    matrix, labels = confusion_matrix(y_true, y_pred, labels)

    per_kelas = {}
    for label in labels:
        precision, recall, f1 = precision_recall_f1(y_true, y_pred, label)
        per_kelas[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for a in y_true if a == label),
        }

    # Macro average memberi bobot sama ke tiap kelas, sehingga kelas minoritas
    # tidak tenggelam. Penting di sini karena kelas earthquake hampir pasti
    # lebih sedikit daripada noise.
    macro_f1 = sum(v["f1"] for v in per_kelas.values()) / len(per_kelas) if per_kelas else 0.0

    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1,
        "false_positive_rate": false_positive_rate(y_true, y_pred, positive_label),
        "positive_label": positive_label,
        "per_class": per_kelas,
        "labels": labels,
        "confusion_matrix": {f"{a}->{p}": n for (a, p), n in matrix.items()},
        "n_samples": len(y_true),
    }


def format_report(report, judul="Laporan"):
    """Render laporan menjadi teks untuk terminal dan dokumentasi."""
    baris = [f"\n=== {judul} ===", f"  Sampel      : {report['n_samples']}"]
    baris.append(f"  Accuracy    : {report['accuracy']:.4f}")
    baris.append(f"  Macro F1    : {report['macro_f1']:.4f}")
    baris.append(
        f"  FP rate     : {report['false_positive_rate']:.4f} "
        f"(kelas '{report['positive_label']}')"
    )
    baris.append(f"\n  {'kelas':<14}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>7}")
    for label in report["labels"]:
        m = report["per_class"][label]
        baris.append(
            f"  {label:<14}{m['precision']:>8.4f}{m['recall']:>8.4f}"
            f"{m['f1']:>8.4f}{m['support']:>7}"
        )

    baris.append("\n  Confusion matrix (baris = asli, kolom = prediksi):")
    header = "  " + " " * 14 + "".join(f"{l:>14}" for l in report["labels"])
    baris.append(header)
    for asli in report["labels"]:
        sel = "".join(
            f"{report['confusion_matrix'].get(f'{asli}->{pred}', 0):>14}"
            for pred in report["labels"]
        )
        baris.append(f"  {asli:<14}{sel}")
    return "\n".join(baris)


def label_distribution(labels):
    return dict(Counter(labels))
