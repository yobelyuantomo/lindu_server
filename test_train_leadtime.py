"""Uji ml_training/train_leadtime.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training.train_leadtime import (  # noqa: E402
    LEAD_WINDOW_S,
    build_events,
    evaluate_baseline,
)


def baris(ts, pga, label="earthquake", node_id="NODE_A", session_id="S001"):
    return {
        "node_id": node_id, "session_id": session_id, "label": label,
        "sensor_ts": ts, "pga": pga, "sta_lta": 3.5, "freq_hz": 8.0,
        "accel_x": 0.6, "accel_y": 0.4, "accel_z": 0.1,
        "temperature": 28.0, "pressure": 1010.0, "gas_raw": 100.0,
    }


def kejadian(session_id="S001", label="earthquake", puncak_pada=3.0, node_id="NODE_A"):
    """Amplitudo naik perlahan lalu memuncak di detik `puncak_pada`."""
    rows = []
    for i in range(50):  # 5 detik @ 10 Hz
        t = i * 0.1
        pga = 0.3 if t < 1.0 else (0.9 if abs(t - puncak_pada) < 0.05 else 0.4)
        rows.append(baris(t, pga, label, node_id, session_id))
    return rows


class TestBuildEvents(unittest.TestCase):
    def test_lead_time_diukur_dari_akhir_jendela(self):
        """Bukan dari awal jendela.

        Model baru bisa memutuskan setelah jendela lengkap. Mengukur dari awal
        jendela akan melebih-lebihkan waktu yang sebenarnya tersedia — persis
        angka yang tidak boleh dilebih-lebihkan pada sistem peringatan dini.
        """
        events = build_events(kejadian(puncak_pada=3.0))
        self.assertEqual(len(events), 1)
        # Jendela awal 1 detik berakhir di t=0.9 (sampel terakhir < 1.0).
        # Puncak di t=3.0, jadi lead ~= 2.1 detik, bukan 3.0.
        self.assertAlmostEqual(events[0]["lead_time"], 2.1, places=5)

    def test_hanya_kelas_earthquake(self):
        rows = kejadian(label="noise")
        self.assertEqual(build_events(rows), [])

    def test_puncak_yang_sudah_lewat_dibuang(self):
        """Puncak sebelum model sempat memutuskan tidak bisa diperingatkan."""
        rows = []
        for i in range(50):
            t = i * 0.1
            pga = 0.9 if t < 0.3 else 0.4   # puncak di awal sekali
            rows.append(baris(t, pga))
        self.assertEqual(build_events(rows), [])

    def test_kejadian_tidak_terpicu_dibuang(self):
        rows = [baris(i * 0.1, 0.01) for i in range(50)]
        self.assertEqual(build_events(rows), [])

    def test_beberapa_sesi(self):
        rows = kejadian("S001", node_id="NODE_A") + kejadian("S002", node_id="NODE_B")
        self.assertEqual(len(build_events(rows)), 2)

    def test_lead_window_mempengaruhi_hasil(self):
        rows = kejadian(puncak_pada=3.0)
        sempit = build_events(rows, lead_window_s=1.0)[0]["lead_time"]
        lebar = build_events(rows, lead_window_s=2.0)[0]["lead_time"]
        self.assertLess(lebar, sempit, "jendela lebih lebar menyisakan lead lebih pendek")

    def test_default_lead_window_satu_detik(self):
        self.assertEqual(LEAD_WINDOW_S, 1.0)


class TestEvaluateBaseline(unittest.TestCase):
    def test_tebakan_konstan(self):
        events = [{"lead_time": 2.0}, {"lead_time": 4.0}]
        hasil = evaluate_baseline(events)
        self.assertAlmostEqual(hasil["konstanta"], 3.0)
        self.assertAlmostEqual(hasil["mae"], 1.0)

    def test_lead_time_seragam_mae_nol(self):
        events = [{"lead_time": 3.0}] * 5
        self.assertAlmostEqual(evaluate_baseline(events)["mae"], 0.0)

    def test_kosong(self):
        self.assertIsNone(evaluate_baseline([]))


if __name__ == "__main__":
    unittest.main()
