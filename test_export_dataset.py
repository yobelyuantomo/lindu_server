"""Uji ml_training/export_dataset.py tanpa memerlukan database."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.export_dataset import (  # noqa: E402
    EXPORT_COLUMNS,
    annotate_row,
    build_query,
    write_csv,
)


class TestBuildQuery(unittest.TestCase):
    def test_filter_freq_hz_selalu_ada(self):
        """Baris tanpa freq_hz tidak boleh ikut terekspor dalam kondisi apa pun."""
        query, _ = build_query()
        self.assertIn("freq_hz IS NOT NULL", query)

    def test_tanpa_filter_tambahan(self):
        query, params = build_query()
        self.assertEqual(params, [])
        self.assertNotIn("time >=", query)

    def test_filter_waktu_dan_node(self):
        query, params = build_query(since="2026-09-01", until="2026-09-30", nodes=["A", "B"])
        self.assertIn("time >= %s", query)
        self.assertIn("time < %s", query)
        self.assertIn("node_id = ANY(%s)", query)
        self.assertEqual(params, ["2026-09-01", "2026-09-30", ["A", "B"]])

    def test_urutan_stabil_per_node_dan_waktu(self):
        query, _ = build_query()
        self.assertIn("ORDER BY node_id", query)


class TestAnnotateRow(unittest.TestCase):
    def test_passed_rule_true(self):
        row = {"pga": 0.5, "sta_lta": 3.0, "freq_hz": 8}
        self.assertTrue(annotate_row(row)["passed_rule"])

    def test_passed_rule_false_karena_frekuensi_tinggi(self):
        row = {"pga": 0.5, "sta_lta": 3.0, "freq_hz": 35}
        self.assertFalse(annotate_row(row)["passed_rule"])

    def test_tidak_mengubah_baris_asli(self):
        row = {"pga": 0.5, "sta_lta": 3.0, "freq_hz": 8}
        annotate_row(row)
        self.assertNotIn("passed_rule", row)


class TestWriteCsv(unittest.TestCase):
    def test_header_dan_isi(self):
        rows = [
            {
                "sensor_ts": 1700000000.5, "recv_time": "2026-09-25T10:00:00Z",
                "node_id": "NODE_A", "pga": 0.5, "sta_lta": 3.0, "freq_hz": 8,
                "accel_x": 0.1, "accel_y": 0.2, "accel_z": 0.3,
                "temperature": 28.0, "pressure": 1010.0, "humidity": 60.0,
                "gas_raw": 120, "gas_alert": False, "valve_status": "OPEN",
            },
            {
                "sensor_ts": 1700000001.5, "recv_time": "2026-09-25T10:00:01Z",
                "node_id": "NODE_A", "pga": 0.01, "sta_lta": 1.0, "freq_hz": 30,
                "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0,
                "temperature": 28.0, "pressure": 1010.0, "humidity": 60.0,
                "gas_raw": 118, "gas_alert": False, "valve_status": "OPEN",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "out.csv")
            written = write_csv(rows, path)
            self.assertEqual(written, 2)

            with open(path, newline="", encoding="utf-8") as handle:
                hasil = list(csv.DictReader(handle))

        self.assertEqual(list(hasil[0].keys()), EXPORT_COLUMNS)
        self.assertEqual(hasil[0]["passed_rule"], "True")
        self.assertEqual(hasil[1]["passed_rule"], "False")
        self.assertEqual(hasil[0]["node_id"], "NODE_A")

    def test_baris_kosong_tetap_menulis_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "kosong.csv")
            self.assertEqual(write_csv([], path), 0)
            with open(path, newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), EXPORT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
