"""Uji ml_training/build_dataset.py (jendela, split per kejadian, augmentasi)."""

import os
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.feature_extractor import FEATURE_NAMES  # noqa: E402
from ml_training.build_dataset import (  # noqa: E402
    META_COLUMNS,
    augment_window,
    build_meta,
    build_rows,
    split_sessions,
    windows_from_rows,
)


def row(node_id="NODE_A", ts=0.0, pga=0.5, label="noise", session_id="S001"):
    return {
        "node_id": node_id, "sensor_ts": ts, "pga": pga, "sta_lta": 3.0,
        "freq_hz": 8.0, "accel_x": 0.3, "accel_y": 0.2, "accel_z": 0.1,
        "temperature": 28.0, "pressure": 1010.0, "humidity": 60.0, "gas_raw": 100.0,
        "label": label, "session_id": session_id, "scenario": "uji",
    }


def sesi_rows(session_id, label, mulai, jumlah=20, node_id="NODE_A", pga=0.5):
    return [
        row(node_id=node_id, ts=mulai + i * 0.1, pga=pga, label=label, session_id=session_id)
        for i in range(jumlah)
    ]


class TestWindowsFromRows(unittest.TestCase):
    def test_jendela_terbentuk_per_durasi(self):
        rows = sesi_rows("S001", "noise", 100.0, jumlah=40)  # 4 detik @10 Hz
        windows = windows_from_rows(rows, window_seconds=2.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["label"], "noise")
        self.assertEqual(windows[0]["session_id"], "S001")

    def test_jendela_menyeberang_sesi_dibuang(self):
        rows = sesi_rows("S001", "noise", 100.0, jumlah=10)
        rows += sesi_rows("S002", "earthquake", 101.0, jumlah=10)
        self.assertEqual(windows_from_rows(rows, window_seconds=2.0), [])

    def test_jendela_tidak_terpicu_dibuang(self):
        rows = sesi_rows("S001", "noise", 100.0, jumlah=40, pga=0.01)
        self.assertEqual(windows_from_rows(rows, window_seconds=2.0), [])

    def test_node_dipisah(self):
        rows = sesi_rows("S001", "noise", 100.0, jumlah=20, node_id="NODE_A")
        rows += sesi_rows("S002", "earthquake", 100.0, jumlah=20, node_id="NODE_B")
        windows = windows_from_rows(rows, window_seconds=2.0)
        self.assertEqual({w["node_id"] for w in windows}, {"NODE_A", "NODE_B"})


class TestSplitSessions(unittest.TestCase):
    def _windows(self, n_per_label=10):
        windows = []
        for label in ("noise", "earthquake"):
            for i in range(n_per_label):
                windows.append({
                    "rows": [], "label": label,
                    "session_id": f"{label[:1].upper()}{i:03d}",
                    "scenario": "uji", "node_id": "NODE_A",
                })
        return windows

    def test_satu_sesi_hanya_masuk_satu_split(self):
        """Inti dari split per kejadian: tidak ada sesi yang terbelah."""
        assignment = split_sessions(self._windows())
        self.assertEqual(len(assignment), 20)
        self.assertTrue(set(assignment.values()) <= {"train", "val", "test"})

    def test_deterministik_dengan_seed_sama(self):
        w = self._windows()
        self.assertEqual(split_sessions(w, seed=7), split_sessions(w, seed=7))

    def test_seed_berbeda_menghasilkan_pembagian_berbeda(self):
        w = self._windows(n_per_label=20)
        self.assertNotEqual(split_sessions(w, seed=1), split_sessions(w, seed=2))

    def test_kedua_kelas_hadir_di_train(self):
        assignment = split_sessions(self._windows())
        label_of = {w["session_id"]: w["label"] for w in self._windows()}
        train_labels = {label_of[s] for s, sp in assignment.items() if sp == "train"}
        self.assertEqual(train_labels, {"noise", "earthquake"})

    def test_sesi_sangat_sedikit_tidak_crash(self):
        windows = [{
            "rows": [], "label": "noise", "session_id": "S001",
            "scenario": "uji", "node_id": "NODE_A",
        }]
        assignment = split_sessions(windows)
        self.assertEqual(assignment, {"S001": "train"})

    def test_test_split_tidak_pernah_kosong_bila_sesi_memadai(self):
        """Regresi: round(5*0.15) menyisakan 0 sesi untuk test.

        Tanpa penjagaan ini, dataset berukuran wajar bisa berakhir tanpa data
        held-out sama sekali dan metrik ujinya tidak ada artinya.
        """
        for n in range(3, 21):
            windows = [{
                "rows": [], "label": "noise", "session_id": f"S{i:03d}",
                "scenario": "uji", "node_id": "NODE_A",
            } for i in range(n)]
            splits = Counter(split_sessions(windows).values())
            self.assertGreaterEqual(splits["test"], 1, f"test kosong saat n={n}")
            self.assertGreaterEqual(splits["val"], 1, f"val kosong saat n={n}")
            self.assertGreaterEqual(splits["train"], 1, f"train kosong saat n={n}")

    def test_dua_sesi_dibagi_latih_dan_uji(self):
        windows = [{
            "rows": [], "label": "noise", "session_id": f"S{i:03d}",
            "scenario": "uji", "node_id": "NODE_A",
        } for i in range(2)]
        self.assertEqual(Counter(split_sessions(windows).values()), Counter({"train": 1, "test": 1}))


class TestAugmentWindow(unittest.TestCase):
    def test_mengubah_nilai_tapi_tidak_merusak_struktur(self):
        rows = sesi_rows("S001", "noise", 100.0, jumlah=10)
        hasil = augment_window(rows, random.Random(1))
        self.assertEqual(len(hasil), len(rows))
        self.assertNotEqual([r["pga"] for r in hasil], [r["pga"] for r in rows])
        self.assertEqual([r["sensor_ts"] for r in hasil], [r["sensor_ts"] for r in rows])

    def test_tidak_mengubah_baris_asli(self):
        rows = sesi_rows("S001", "noise", 100.0, jumlah=5)
        asli = [r["pga"] for r in rows]
        augment_window(rows, random.Random(1))
        self.assertEqual([r["pga"] for r in rows], asli)

    def test_label_dan_sesi_ikut_terbawa(self):
        rows = sesi_rows("S007", "earthquake", 100.0, jumlah=5)
        hasil = augment_window(rows, random.Random(1))
        self.assertTrue(all(r["session_id"] == "S007" for r in hasil))


class TestBuildRows(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(6):
            rows += sesi_rows(f"N{i:03d}", "noise", 100.0 + i * 10, jumlah=20)
        for i in range(6):
            rows += sesi_rows(f"E{i:03d}", "earthquake", 500.0 + i * 10, jumlah=20)
        self.windows = windows_from_rows(rows, window_seconds=2.0)
        self.assignment = split_sessions(self.windows)

    def test_augmentasi_hanya_di_train(self):
        dataset = build_rows(self.windows, self.assignment, augment_factor=2)
        for baris in dataset:
            if baris["is_augmented"]:
                self.assertEqual(baris["split"], "train")

    def test_augment_factor_nol(self):
        dataset = build_rows(self.windows, self.assignment, augment_factor=0)
        self.assertFalse(any(b["is_augmented"] for b in dataset))
        self.assertEqual(len(dataset), len(self.windows))

    def test_kolom_lengkap(self):
        dataset = build_rows(self.windows, self.assignment, augment_factor=1)
        for kolom in META_COLUMNS + FEATURE_NAMES:
            self.assertIn(kolom, dataset[0])

    def test_sesi_tidak_bocor_antar_split(self):
        """Satu session_id tidak boleh muncul di lebih dari satu split."""
        dataset = build_rows(self.windows, self.assignment, augment_factor=2)
        split_per_sesi = {}
        for baris in dataset:
            split_per_sesi.setdefault(baris["session_id"], set()).add(baris["split"])
        for session_id, splits in split_per_sesi.items():
            self.assertEqual(len(splits), 1, f"{session_id} bocor ke {splits}")


class TestBuildMeta(unittest.TestCase):
    def test_isi_metadata(self):
        rows = sesi_rows("S001", "noise", 100.0, jumlah=40)
        windows = windows_from_rows(rows, window_seconds=2.0)
        assignment = split_sessions(windows)
        dataset = build_rows(windows, assignment, augment_factor=1)
        meta = build_meta(dataset, assignment, 42, (0.7, 0.15, 0.15), 2.0, 1)

        self.assertEqual(meta["split_unit"], "session_id")
        self.assertEqual(meta["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(meta["n_features"], len(FEATURE_NAMES))
        self.assertEqual(meta["n_rows"], len(dataset))
        self.assertEqual(meta["seed"], 42)


if __name__ == "__main__":
    unittest.main()
