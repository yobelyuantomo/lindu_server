"""Uji ml_training/record_session.py (pencatat sesi perekaman)."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.record_session import (  # noqa: E402
    SESSION_COLUMNS,
    append_session,
    find_overlap,
    load_sessions,
)


def sesi(node_id="NODE_A", label="noise", scenario="melompat", start=100.0, end=200.0, notes=""):
    return {
        "node_id": node_id, "label": label, "scenario": scenario,
        "start_ts": f"{start:.3f}", "end_ts": f"{end:.3f}", "notes": notes,
    }


class TestFindOverlap(unittest.TestCase):
    def setUp(self):
        self.sessions = [sesi(start=100.0, end=200.0)]

    def test_tidak_bertindihan_sebelum(self):
        self.assertIsNone(find_overlap(self.sessions, "NODE_A", 50.0, 99.0))

    def test_tidak_bertindihan_sesudah(self):
        self.assertIsNone(find_overlap(self.sessions, "NODE_A", 201.0, 300.0))

    def test_bersentuhan_di_batas_bukan_bertindihan(self):
        self.assertIsNone(find_overlap(self.sessions, "NODE_A", 200.0, 300.0))
        self.assertIsNone(find_overlap(self.sessions, "NODE_A", 50.0, 100.0))

    def test_bertindihan_sebagian(self):
        self.assertIsNotNone(find_overlap(self.sessions, "NODE_A", 150.0, 250.0))

    def test_bertindihan_seluruhnya(self):
        self.assertIsNotNone(find_overlap(self.sessions, "NODE_A", 50.0, 300.0))

    def test_node_berbeda_boleh_bersamaan(self):
        """Dua node memang bisa merekam skenario berbeda pada saat yang sama."""
        self.assertIsNone(find_overlap(self.sessions, "NODE_B", 150.0, 250.0))


class TestSessionsFile(unittest.TestCase):
    def test_berkas_belum_ada_dianggap_kosong(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_sessions(os.path.join(tmp, "belum_ada.csv")), [])

    def test_append_menulis_header_sekali(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sessions.csv")
            append_session(path, sesi(scenario="berjalan", start=10.0, end=20.0))
            append_session(path, sesi(scenario="melompat", start=30.0, end=40.0))

            with open(path, newline="", encoding="utf-8") as handle:
                baris = list(csv.reader(handle))
            terbaca = load_sessions(path)

        self.assertEqual(baris[0], SESSION_COLUMNS)
        self.assertEqual(len(baris), 3)  # header + 2 sesi
        self.assertEqual(terbaca[1]["scenario"], "melompat")

    def test_round_trip_deteksi_bentrok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sessions.csv")
            append_session(path, sesi(start=100.0, end=200.0))
            tersimpan = load_sessions(path)
            self.assertIsNotNone(find_overlap(tersimpan, "NODE_A", 150.0, 250.0))


if __name__ == "__main__":
    unittest.main()
