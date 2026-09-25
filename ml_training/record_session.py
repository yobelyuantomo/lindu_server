"""Pencatat sesi perekaman dataset.

Pelabelan bekerja dengan mencocokkan rentang waktu, sehingga waktu mulai dan
selesai tiap sesi harus tercatat dalam epoch. Mengonversi jam dinding ke epoch
secara manual mudah sekali meleset, dan label yang meleset merusak dataset
tanpa gejala yang kelihatan — karena itu pencatatannya diotomasi di sini.

Pemakaian::

    python -m ml_training.record_session --node NODE_A --label noise \\
        --scenario "membanting pintu"

Tekan ENTER untuk mulai, ENTER lagi untuk selesai. Sesi ditambahkan ke
``ml_training/sessions.csv``.

Lihat ``RECORDING_PROTOCOL.md`` untuk daftar skenario dan aturan kebersihan data.
"""

import argparse
import csv
import os
import time
from datetime import datetime

VALID_LABELS = ("earthquake", "noise", "gas_leak")

SESSION_COLUMNS = ["node_id", "label", "scenario", "start_ts", "end_ts", "notes"]

DEFAULT_SESSIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.csv")


def load_sessions(path):
    """Baca sesi yang sudah tercatat. Berkas belum ada dianggap kosong."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_overlap(sessions, node_id, start_ts, end_ts):
    """Cari sesi yang rentang waktunya bertindihan pada node yang sama.

    Rentang bertindihan membuat satu baris telemetri punya dua label, dan
    ``label_dataset.py`` tidak punya cara memilih mana yang benar. Lebih baik
    ditolak saat pencatatan, selagi pelakunya masih ingat apa yang terjadi.
    """
    for row in sessions:
        if row["node_id"] != node_id:
            continue
        existing_start = float(row["start_ts"])
        existing_end = float(row["end_ts"])
        if start_ts < existing_end and end_ts > existing_start:
            return row
    return None


def append_session(path, session):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SESSION_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(session)


def _fmt(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, help="node_id yang direkam")
    parser.add_argument("--label", required=True, choices=VALID_LABELS)
    parser.add_argument("--scenario", required=True, help='mis. "melompat"')
    parser.add_argument("--notes", default="", help="kondisi tidak biasa saat merekam")
    parser.add_argument("--sessions", default=DEFAULT_SESSIONS_PATH)
    args = parser.parse_args(argv)

    print(f"\nNode     : {args.node}")
    print(f"Label    : {args.label}")
    print(f"Skenario : {args.scenario}")
    print("\nIngat: beri jeda ~5 detik sebelum mulai beraktivitas, supaya")
    print("getaran dari tangan yang menekan tombol tidak ikut terlabeli.")
    print("Pantau panel PGA di Grafana — kalau tidak pernah lewat 0.12 G,")
    print("aktivitasnya kurang kuat dan datanya tidak akan terpakai.\n")

    input("Tekan ENTER untuk MULAI...")
    start_ts = time.time()
    print(f"  mulai   {_fmt(start_ts)}")

    input("Tekan ENTER untuk SELESAI...")
    end_ts = time.time()
    print(f"  selesai {_fmt(end_ts)}  ({end_ts - start_ts:.1f} detik)")

    sessions = load_sessions(args.sessions)
    bentrok = find_overlap(sessions, args.node, start_ts, end_ts)
    if bentrok:
        print(
            f"\n[!] DITOLAK: rentang bertindihan dengan sesi "
            f"'{bentrok['scenario']}' ({bentrok['label']}) pada node yang sama."
        )
        print("    Satu baris telemetri tidak boleh punya dua label.")
        return 1

    durasi = end_ts - start_ts
    if durasi < 10:
        print(f"\n[!] Sesi hanya {durasi:.1f} detik — kemungkinan besar terlalu")
        print("    pendek untuk menghasilkan jendela yang cukup (1 jendela = 2 detik).")

    append_session(
        args.sessions,
        {
            "node_id": args.node,
            "label": args.label,
            "scenario": args.scenario,
            "start_ts": f"{start_ts:.3f}",
            "end_ts": f"{end_ts:.3f}",
            "notes": args.notes,
        },
    )
    print(f"\n[OK] Sesi dicatat ke {args.sessions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
