"""Beri label pada telemetri hasil ekspor berdasarkan sesi perekaman.

Pelabelan bekerja dengan mencocokkan **rentang waktu**: satu baris telemetri
mendapat label sesi yang rentangnya memuat ``sensor_ts`` baris tersebut pada
node yang sama. Sumber kebenarannya adalah ``sessions.csv`` yang dihasilkan
``record_session.py`` — jangan pernah melabeli manual di CSV telemetri.

Pemakaian::

    python -m ml_training.label_dataset \\
        --telemetry data/raw/telemetry.csv --out data/raw/labeled.csv

    # hanya melihat ringkasan tanpa menulis berkas
    python -m ml_training.label_dataset --telemetry data/raw/cek.csv --summary

Baris yang tidak masuk rentang sesi mana pun **dibuang**. Itu adalah telemetri
latar yang tidak diketahui isinya — memasukkannya sebagai ``noise`` akan
mencemari kelas negatif dengan data yang tidak pernah diverifikasi siapa pun.
"""

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import (  # noqa: E402
    build_windows,
    is_analyzable_window,
)
from ml_training.record_session import DEFAULT_SESSIONS_PATH, load_sessions  # noqa: E402

#: Kolom numerik yang perlu dikonversi dari string CSV.
_FLOAT_COLUMNS = (
    "sensor_ts", "pga", "sta_lta", "freq_hz",
    "accel_x", "accel_y", "accel_z",
    "temperature", "pressure", "humidity", "gas_raw",
)


def parse_row(raw):
    """Konversi satu baris CSV menjadi dict bertipe benar.

    Nilai kosong menjadi ``None``, bukan 0. Membedakan "tidak terkirim" dari
    "bernilai nol" itu penting: ``freq_hz`` bernilai 0 lolos filter rule-based
    sementara ``None`` tidak.
    """
    row = dict(raw)
    for col in _FLOAT_COLUMNS:
        value = row.get(col)
        if value is None or value == "":
            row[col] = None
        else:
            try:
                row[col] = float(value)
            except (TypeError, ValueError):
                row[col] = None
    return row


def build_session_index(sessions):
    """Kelompokkan sesi per node agar pencarian tidak menyapu seluruh daftar.

    Setiap sesi diberi ``session_id`` stabil berdasarkan urutan (node, waktu
    mulai). Id ini dipakai ``build_dataset.py`` untuk memecah data per
    kejadian, sehingga jendela dari satu sesi yang sama tidak pernah tersebar
    ke train dan test sekaligus.
    """
    index = {}
    urut = sorted(sessions, key=lambda s: (s["node_id"], float(s["start_ts"])))
    for nomor, s in enumerate(urut, start=1):
        index.setdefault(s["node_id"], []).append(
            {
                "session_id": f"S{nomor:03d}",
                "label": s["label"],
                "scenario": s.get("scenario", ""),
                "start_ts": float(s["start_ts"]),
                "end_ts": float(s["end_ts"]),
            }
        )
    for node_sessions in index.values():
        node_sessions.sort(key=lambda s: s["start_ts"])
    return index


def session_for(index, node_id, sensor_ts):
    """Cari sesi yang memuat satu baris. ``None`` bila di luar seluruh sesi."""
    if sensor_ts is None:
        return None
    for s in index.get(node_id, ()):
        if s["start_ts"] <= sensor_ts < s["end_ts"]:
            return s
    return None


def label_for(index, node_id, sensor_ts):
    """Cari label untuk satu baris. ``None`` bila di luar seluruh sesi."""
    sesi = session_for(index, node_id, sensor_ts)
    return sesi["label"] if sesi else None


def label_rows(rows, index):
    """Beri label pada baris yang masuk rentang sesi; sisanya dibuang.

    Mengembalikan ``(baris_berlabel, jumlah_terbuang)``.
    """
    labeled, dibuang = [], 0
    for row in rows:
        sesi = session_for(index, row.get("node_id"), row.get("sensor_ts"))
        if sesi is None:
            dibuang += 1
            continue
        out = dict(row)
        out["label"] = sesi["label"]
        out["session_id"] = sesi["session_id"]
        out["scenario"] = sesi["scenario"]
        labeled.append(out)
    return labeled, dibuang


def summarize_windows(labeled_rows, window_seconds=2.0):
    """Hitung jendela yang benar-benar layak dianalisis, per label.

    Ini angka yang menentukan apakah dataset sudah cukup — bukan jumlah baris
    mentah. Jendela yang tidak terpicu tidak dihitung karena tidak akan ikut
    ke dalam dataset latih.
    """
    per_node = {}
    for row in labeled_rows:
        per_node.setdefault(row["node_id"], []).append(row)

    layak = Counter()
    tidak_terpicu = Counter()
    for node_rows in per_node.values():
        node_rows.sort(key=lambda r: (r.get("sensor_ts") or 0.0))
        for window in build_windows(node_rows, window_seconds):
            labels = {r["label"] for r in window}
            if len(labels) != 1:
                # Jendela menyeberangi batas dua sesi berbeda: dibuang karena
                # labelnya ambigu.
                continue
            label = labels.pop()
            if is_analyzable_window(window):
                layak[label] += 1
            else:
                tidak_terpicu[label] += 1
    return layak, tidak_terpicu


def write_labeled(rows, path, fieldnames):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return len(rows)


def _print_summary(labeled, dibuang, layak, tidak_terpicu):
    total_layak = sum(layak.values())
    print(f"\n  Baris berlabel   : {len(labeled)}")
    print(f"  Baris dibuang    : {dibuang} (di luar seluruh rentang sesi)")
    print(f"\n  Jendela LAYAK dianalisis (terpicu, PGA >= 0.12):")
    if not layak:
        print("      (tidak ada)")
    for label, jumlah in sorted(layak.items()):
        print(f"      {label:<12} {jumlah}")
    print(f"      {'TOTAL':<12} {total_layak}")

    if tidak_terpicu:
        print(f"\n  Jendela terbuang karena tidak terpicu:")
        for label, jumlah in sorted(tidak_terpicu.items()):
            print(f"      {label:<12} {jumlah}")
        print(
            "      -> Getaran saat merekam kurang kuat. Lihat RECORDING_PROTOCOL.md:\n"
            "         aktivitas harus melewati PGA 0.12 agar datanya terpakai."
        )

    if total_layak == 0:
        print(
            "\n[!] Tidak ada satu pun jendela yang layak. Dataset belum bisa dibangun."
        )
    elif len(layak) < 2:
        print(
            "\n[!] Hanya ada satu kelas. Classifier butuh minimal dua kelas "
            "(earthquake dan noise)."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, help="CSV hasil export_dataset.py")
    parser.add_argument("--sessions", default=DEFAULT_SESSIONS_PATH)
    parser.add_argument("--out", help="CSV keluaran berlabel (wajib kecuali --summary)")
    parser.add_argument("--summary", action="store_true", help="hanya cetak ringkasan")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    if not args.summary and not args.out:
        parser.error("--out wajib diisi kecuali memakai --summary")

    sessions = load_sessions(args.sessions)
    if not sessions:
        print(f"[!] {args.sessions} kosong atau tidak ada. Belum ada sesi perekaman.")
        return 1
    index = build_session_index(sessions)
    print(f"[OK] {len(sessions)} sesi dimuat dari {args.sessions}")

    with open(args.telemetry, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [parse_row(r) for r in reader]
    print(f"[OK] {len(rows)} baris telemetri dibaca dari {args.telemetry}")

    labeled, dibuang = label_rows(rows, index)
    layak, tidak_terpicu = summarize_windows(labeled, args.window_seconds)
    _print_summary(labeled, dibuang, layak, tidak_terpicu)

    if not args.summary:
        jumlah = write_labeled(
            labeled, args.out, fieldnames + ["label", "session_id", "scenario"]
        )
        print(f"\n[OK] {jumlah} baris berlabel ditulis ke {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
