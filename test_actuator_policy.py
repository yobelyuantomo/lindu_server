"""Uji ml/actuator_policy.py.

Aktuator di sini mengunci katup gas dan membuka pintu darurat. Yang paling
penting dibuktikan adalah bahwa lapisan ML tidak pernah bisa MELEMAHKAN
respons terhadap gempa yang sudah tervalidasi fisika.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.actuator_policy import (  # noqa: E402
    BAHAYA,
    KRITIS,
    NORMAL,
    SIAGA,
    WASPADA,
    decide,
    tingkat_tertinggi,
    to_mqtt_level,
)


def ml(label="earthquake", confidence=0.95, available=True, over_budget=False,
       lead_time=None, magnitude=None):
    return {
        "available": available, "label": label, "confidence": confidence,
        "over_budget": over_budget, "lead_time_pred": lead_time,
        "magnitude_pred": magnitude,
    }


class TestTingkatTertinggi(unittest.TestCase):
    def test_urutan(self):
        self.assertEqual(tingkat_tertinggi(NORMAL, BAHAYA), BAHAYA)
        self.assertEqual(tingkat_tertinggi(KRITIS, SIAGA), KRITIS)
        self.assertEqual(tingkat_tertinggi(WASPADA, WASPADA), WASPADA)


class TestKeselamatanRuleBased(unittest.TestCase):
    """Aturan paling penting: ML tidak boleh menurunkan respons rule-based."""

    def test_konsensus_selalu_minimal_bahaya(self):
        for hasil_ml in (None, ml(label="noise"), ml(confidence=0.0),
                         ml(available=False), ml(over_budget=True)):
            hasil = decide(rule_consensus=True, ml_result=hasil_ml)
            self.assertEqual(hasil["level"], BAHAYA)
            self.assertTrue(hasil["actions"]["siren"])
            self.assertTrue(hasil["actions"]["lock_valve"])

    def test_shadow_mode_tidak_melemahkan_rule_based(self):
        hasil = decide(rule_consensus=True, ml_result=ml(), shadow_mode=True)
        self.assertEqual(hasil["level"], BAHAYA)
        self.assertTrue(hasil["actions"]["siren"])

    def test_ml_boleh_menaikkan_ke_kritis(self):
        hasil = decide(rule_consensus=True, ml_result=ml(confidence=0.95, magnitude=7.0))
        self.assertEqual(hasil["level"], KRITIS)
        self.assertTrue(hasil["actions"]["persist_nvs"])


class TestTingkatDariMl(unittest.TestCase):
    def test_confidence_memetakan_tingkat(self):
        kasus = [(0.30, NORMAL), (0.55, SIAGA), (0.80, WASPADA), (0.95, BAHAYA)]
        for confidence, diharapkan in kasus:
            hasil = decide(rule_consensus=False, ml_result=ml(confidence=confidence))
            self.assertEqual(hasil["level"], diharapkan, f"confidence={confidence}")

    def test_tepat_di_ambang(self):
        self.assertEqual(decide(False, ml(confidence=0.50))["level"], SIAGA)
        self.assertEqual(decide(False, ml(confidence=0.75))["level"], WASPADA)
        self.assertEqual(decide(False, ml(confidence=0.90))["level"], BAHAYA)

    def test_label_bukan_gempa_diabaikan(self):
        hasil = decide(rule_consensus=False, ml_result=ml(label="noise", confidence=0.99))
        self.assertEqual(hasil["level"], NORMAL)

    def test_over_budget_diabaikan_dan_dicatat(self):
        hasil = decide(rule_consensus=False, ml_result=ml(over_budget=True))
        self.assertEqual(hasil["level"], NORMAL)
        self.assertIn("latensi", hasil["reason"])

    def test_ml_tidak_tersedia(self):
        self.assertEqual(decide(False, None)["level"], NORMAL)
        self.assertEqual(decide(False, ml(available=False))["level"], NORMAL)

    def test_lead_time_pendek_langsung_bahaya(self):
        """Kalau guncangan kuat tiba dalam hitungan detik, menunggu keyakinan
        model bertambah tidak ada gunanya."""
        hasil = decide(rule_consensus=False, ml_result=ml(confidence=0.55, lead_time=3.0))
        self.assertEqual(hasil["level"], BAHAYA)

    def test_lead_time_panjang_tidak_menaikkan(self):
        hasil = decide(rule_consensus=False, ml_result=ml(confidence=0.55, lead_time=30.0))
        self.assertEqual(hasil["level"], SIAGA)

    def test_magnitudo_kecil_tidak_jadi_kritis(self):
        hasil = decide(rule_consensus=False, ml_result=ml(confidence=0.95, magnitude=4.0))
        self.assertEqual(hasil["level"], BAHAYA)


class TestShadowMode(unittest.TestCase):
    def test_wewenang_ml_dipangkas_sampai_siaga(self):
        hasil = decide(rule_consensus=False, ml_result=ml(confidence=0.99), shadow_mode=True)
        self.assertEqual(hasil["level"], SIAGA)
        self.assertFalse(hasil["actions"]["siren"])
        self.assertFalse(hasil["actions"]["lock_valve"])
        self.assertIn("shadow mode", hasil["reason"])

    def test_shadow_mode_tanpa_indikasi_tetap_normal(self):
        hasil = decide(rule_consensus=False, ml_result=ml(label="noise"), shadow_mode=True)
        self.assertEqual(hasil["level"], NORMAL)


class TestAksi(unittest.TestCase):
    def test_tingkat_ringan_tidak_menyentuh_aktuator_fisik(self):
        """Menahan aksi bukan berarti diam: LED tetap menyala."""
        for tingkat_conf, tingkat in ((0.55, SIAGA), (0.80, WASPADA)):
            hasil = decide(rule_consensus=False, ml_result=ml(confidence=tingkat_conf))
            self.assertEqual(hasil["level"], tingkat)
            self.assertFalse(hasil["actions"]["siren"])
            self.assertFalse(hasil["actions"]["lock_valve"])
            self.assertFalse(hasil["actions"]["open_door"])
            self.assertNotEqual(hasil["actions"]["led"], "hijau")

    def test_hanya_kritis_yang_persist_nvs(self):
        for rule, hasil_ml, persist in (
            (True, None, False),
            (True, ml(confidence=0.95, magnitude=7.0), True),
        ):
            self.assertEqual(decide(rule, hasil_ml)["actions"]["persist_nvs"], persist)

    def test_actions_adalah_salinan(self):
        """Pemanggil tidak boleh bisa mengubah tabel AKSI global."""
        hasil = decide(rule_consensus=True)
        hasil["actions"]["siren"] = False
        self.assertTrue(decide(rule_consensus=True)["actions"]["siren"])


class TestToMqttLevel(unittest.TestCase):
    def test_hanya_bahaya_dan_kritis_yang_critical(self):
        """Firmware lama hanya mengenal CRITICAL sebagai pemicu sirine, jadi
        tingkat di bawahnya tidak boleh dipetakan ke sana."""
        self.assertEqual(to_mqtt_level(BAHAYA), "CRITICAL")
        self.assertEqual(to_mqtt_level(KRITIS), "CRITICAL")
        for tingkat in (NORMAL, SIAGA, WASPADA):
            self.assertNotEqual(to_mqtt_level(tingkat), "CRITICAL")


if __name__ == "__main__":
    unittest.main()
