"""Pastikan algoritma seismik di node ESP32 dan ESP8266 tetap identik.

Kedua firmware menghitung PGA, STA/LTA, dan frekuensi dominan dengan kode yang
disalin, bukan dibagi — tidak ada cara membagi sumber antara dua proyek
PlatformIO untuk keluarga chip berbeda tanpa membuat keduanya rapuh.

Salinan bisa menyimpang, dan kalau menyimpang akibatnya tidak terlihat:

- STA/LTA dihitung dengan EMA per sampel; frekuensi dari zero-crossing per
  detik. Keduanya hanya bermakna relatif terhadap laju sampling.
- Node dengan konstanta berbeda melaporkan sta_lta dan freq_hz yang berbeda
  untuk guncangan fisik yang sama.
- Ambang rule-based (0.12 / 2.0 / 20 Hz) menjadi tidak valid untuk node itu,
  dan dataset latih tercemar: classifier belajar membedakan merek chip.

Tidak ada error yang muncul. Metriknya tetap terlihat bagus. Karena itu
perbandingannya dijadikan test.
"""

import os
import re
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
ESP32_SENSOR = os.path.join(_HERE, "..", "esp32_sensor_node", "src", "SensorManager.cpp")
ESP8266_CORE = os.path.join(_HERE, "..", "esp8266_sensor_node", "src", "SeismicCore.h")

FIRMWARE_ADA = os.path.exists(ESP32_SENSOR) and os.path.exists(ESP8266_CORE)


def _baca(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _cari(teks, pola, nama):
    m = re.search(pola, teks)
    if not m:
        raise AssertionError(
            f"Tidak menemukan {nama}. Firmware mungkin sudah diubah strukturnya; "
            "perbarui pola di test ini, jangan hapus pemeriksaannya."
        )
    return float(m.group(1))


@unittest.skipUnless(FIRMWARE_ADA, "sumber firmware tidak tersedia di checkout ini")
class TestKonsistensiKonstantaSeismik(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.esp32 = _baca(ESP32_SENSOR)
        cls.esp8266 = _baca(ESP8266_CORE)

    def test_interval_sampling(self):
        """Paling menentukan: seluruh konstanta lain bermakna relatif padanya."""
        a = _cari(self.esp32, r"millis\(\)\s*-\s*_last_read_time\s*<\s*(\d+)",
                  "interval sampling ESP32")
        b = _cari(self.esp8266, r"SEIS_SAMPLE_INTERVAL_MS\s+(\d+)",
                  "interval sampling ESP8266")
        self.assertEqual(a, b, "laju sampling berbeda -> sta_lta & freq_hz tidak sebanding")

    def test_alpha_dc(self):
        a = _cari(self.esp32, r"alpha_dc\s*=\s*([\d.]+)", "alpha_dc ESP32")
        b = _cari(self.esp8266, r"SEIS_ALPHA_DC\s+([\d.]+)f", "alpha_dc ESP8266")
        self.assertAlmostEqual(a, b, places=6)

    def test_alpha_sta(self):
        a = _cari(self.esp32, r"_sta_ema\s*=\s*\(energy\s*\*\s*([\d.]+)\)", "alpha STA ESP32")
        b = _cari(self.esp8266, r"SEIS_ALPHA_STA\s+([\d.]+)f", "alpha STA ESP8266")
        self.assertAlmostEqual(a, b, places=6)

    def test_alpha_lta(self):
        a = _cari(self.esp32, r"_lta_ema\s*=\s*\(energy\s*\*\s*([\d.]+)\)", "alpha LTA ESP32")
        b = _cari(self.esp8266, r"SEIS_ALPHA_LTA\s+([\d.]+)f", "alpha LTA ESP8266")
        self.assertAlmostEqual(a, b, places=6)

    def test_lantai_lta(self):
        a = _cari(self.esp32, r"_lta_ema\s*<\s*([\d.]+)f", "lantai LTA ESP32")
        b = _cari(self.esp8266, r"SEIS_LTA_FLOOR\s+([\d.]+)f", "lantai LTA ESP8266")
        self.assertAlmostEqual(a, b, places=9)

    def test_jendela_zero_crossing(self):
        a = _cari(self.esp32, r"millis\(\)\s*-\s*_zcr_timer\s*>=\s*(\d+)", "jendela ZCR ESP32")
        b = _cari(self.esp8266, r"SEIS_ZCR_WINDOW_MS\s+(\d+)", "jendela ZCR ESP8266")
        self.assertEqual(a, b)

    def test_ambang_spike(self):
        a = _cari(self.esp32, r"is_earthquake_spike\s*=\s*\(pga\s*>\s*([\d.]+)\)",
                  "ambang spike ESP32")
        b = _cari(self.esp8266, r"SEIS_SPIKE_PGA\s+([\d.]+)f", "ambang spike ESP8266")
        self.assertAlmostEqual(a, b, places=6)

    def test_laju_telemetri(self):
        """Laju telemetri menentukan kepadatan sampel per jendela.

        Jendela dari node yang mengirim lebih rapat akan punya jumlah sampel
        berbeda, dan fitur berbasis integrasi waktu ikut bergeser.
        """
        cepat32 = _cari(self.esp32, r"_telemetry_last_send\s*>=\s*(\d+)\)\s*\|\|",
                        "telemetri cepat ESP32")
        cepat66 = _cari(self.esp8266, r"SEIS_TELEMETRY_FAST_MS\s+(\d+)",
                        "telemetri cepat ESP8266")
        self.assertEqual(cepat32, cepat66)

        lambat32 = _cari(self.esp32, r"\|\|\s*\(millis\(\)\s*-\s*_telemetry_last_send\s*>=\s*(\d+)\)",
                         "telemetri lambat ESP32")
        lambat66 = _cari(self.esp8266, r"SEIS_TELEMETRY_SLOW_MS\s+(\d+)",
                         "telemetri lambat ESP8266")
        self.assertEqual(lambat32, lambat66)

    def test_ambang_pga_sama_dengan_feature_extractor(self):
        """Ambang di firmware harus sama dengan yang dipakai pipeline ML."""
        from ml.feature_extractor import RULE_PGA_MIN

        b = _cari(self.esp8266, r"SEIS_SPIKE_PGA\s+([\d.]+)f", "ambang spike ESP8266")
        self.assertAlmostEqual(b, RULE_PGA_MIN, places=6)


@unittest.skipUnless(FIRMWARE_ADA, "sumber firmware tidak tersedia di checkout ini")
class TestFormulaSeismik(unittest.TestCase):
    """Rumusnya, bukan hanya konstantanya, harus sama bentuknya."""

    @classmethod
    def setUpClass(cls):
        cls.esp32 = _baca(ESP32_SENSOR)
        cls.esp8266 = _baca(
            os.path.join(_HERE, "..", "esp8266_sensor_node", "src", "SeismicCore.cpp")
        )

    def test_pga_dibagi_gravitasi(self):
        for teks, nama in ((self.esp32, "ESP32"), (self.esp8266, "ESP8266")):
            self.assertIn("9.81", teks, f"pembagi gravitasi hilang di {nama}")

    def test_rms_dibagi_tiga(self):
        self.assertRegex(self.esp32, r"/\s*3\.0")
        self.assertRegex(self.esp8266, r"/\s*3\.0f")

    def test_zero_crossing_dibagi_dua(self):
        """Satu siklus penuh melintasi nol dua kali."""
        for teks, nama in ((self.esp32, "ESP32"), (self.esp8266, "ESP8266")):
            self.assertRegex(teks, r"_zcr_count\s*/\s*2", f"pembagi ZCR salah di {nama}")

    def test_energi_adalah_pga_kuadrat(self):
        for teks, nama in ((self.esp32, "ESP32"), (self.esp8266, "ESP8266")):
            self.assertRegex(teks, r"energy\s*=\s*pga\s*\*\s*pga",
                             f"definisi energi berbeda di {nama}")


if __name__ == "__main__":
    unittest.main()
