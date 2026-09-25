# Protokol Perekaman Dataset

Panduan mengumpulkan data latih untuk classifier Assignment 3.

Ini adalah **pekerjaan fisik** yang tidak bisa diotomasi dan biasanya menjadi
jalur terpanjang keseluruhan proyek. Mulai secepatnya, paralel dengan
pengerjaan kode.

---

## Prasyarat

1. **Ingester versi baru sudah ter-deploy.** Tanpa ini `freq_hz` tidak tersimpan
   dan seluruh rekaman menjadi tidak terpakai:

   ```bash
   cd prototype/grafana-stack && docker-compose up -d --build
   ```

   Verifikasi sebelum merekam apa pun:

   ```sql
   SELECT COUNT(*) FROM sensor_telemetry WHERE freq_hz IS NOT NULL;
   ```

   Angkanya harus bertambah dari waktu ke waktu. Kalau tetap nol, jangan mulai
   merekam — datanya akan terbuang.

2. **Jam node tersinkron NTP.** Pelabelan bekerja dengan mencocokkan rentang
   waktu, jadi jam yang meleset merusak seluruh label. Cek di dashboard bahwa
   latensi node wajar (bukan puluhan detik).

3. **Catat node mana yang dipakai.** Baseline kebisingan tiap node berbeda;
   node di dekat jalan tidak sebanding dengan node di ruang tenang.

---

## Kenapa hanya merekam getaran yang cukup kuat

Firmware mengirim telemetri 10 Hz saat PGA > 0,12 dan 1 Hz saat tenang.
Classifier hanya dilatih pada jendela **terpicu** (lihat `is_analyzable_window`
di `ml/feature_extractor.py`), supaya kelas gempa dan non-gempa sama-sama
terekam pada laju yang sama dan model tidak bisa curang dengan menghafal
kepadatan sampel.

Konsekuensi praktis: **setiap sesi perekaman harus benar-benar menggoyang
sensor sampai melewati ambang.** Berjalan pelan di ujung ruangan tidak berguna
sebagai data — tidak akan terpicu, dan barisnya tidak masuk dataset.

Cara memastikan: pantau panel PGA di Grafana saat merekam. Kalau jarumnya tidak
pernah melewati 0,12 G, perkuat aktivitasnya atau dekatkan ke sensor.

---

## Kelas yang dikumpulkan

| Label | Isi | Target minimum |
|---|---|---|
| `noise` | Getaran kuat non-seismik dari aktivitas manusia/lingkungan | ≥ 30 jendela per skenario |
| `earthquake` | Guncangan menyerupai gempa (meja getar / simulasi) | ≥ 100 jendela total |
| `gas_leak` | Khusus node varian Classic dengan MQ-2 | opsional |

Satu jendela = 2 detik. Pada 10 Hz berarti ±20 baris telemetri. Jadi sesi
30 detik yang benar-benar terpicu ≈ 15 jendela.

---

## Skenario kelas `noise` (wajib, minimal 6)

Tujuannya meniru **false positive yang selama ini mengganggu sistem lama** —
getaran kuat berfrekuensi tinggi yang bukan gempa.

| # | Skenario | Cara | Durasi |
|---|---|---|---|
| 1 | Berjalan | Berjalan normal di dekat sensor, bolak-balik | 60 s |
| 2 | Melompat | Melompat di lantai yang sama dengan sensor | 30 s |
| 3 | Memukul meja | Memukul permukaan tempat sensor dipasang | 30 s |
| 4 | Membanting pintu | Buka-tutup pintu dengan keras, berulang | 30 s |
| 5 | Kendaraan lewat | Rekam saat motor/mobil melintas dekat bangunan | 60 s |
| 6 | Konstruksi ringan | Bor, palu, atau memindahkan furnitur | 60 s |

Variasikan jarak sensor (dekat dan jauh) agar model tidak hanya belajar
"amplitudo besar = noise".

---

## Skenario kelas `earthquake`

Kejadian gempa alami hampir pasti nihil selama masa proyek. Dua pengganti:

**A. Meja getar sederhana.** Letakkan node di atas papan, goyang horizontal
secara kontinu dan ritmis (1–15 Hz) selama 30–60 detik. Kunci pembedanya dari
`noise`: gempa bersifat **berkelanjutan, horizontal, dan berfrekuensi rendah**,
sedangkan hentakan bersifat impulsif, vertikal, dan berfrekuensi tinggi.
Goyangan harus mulai pelan, menguat, lalu meluruh — bukan hentakan tunggal.

**B. Simulasi Wokwi.** Injeksikan profil sinyal menyerupai rekaman gempa nyata.
Berguna untuk memperbanyak variasi magnitudo yang tidak mungkin diperagakan
secara fisik.

> **Wajib ditulis di bab Limitasi laporan.** Kelas `earthquake` berasal dari
> peragaan, bukan gempa tektonik sungguhan. Metrik seberapa pun bagusnya tidak
> membuktikan sistem akan bekerja pada gempa nyata. Menyembunyikan hal ini
> justru lebih merugikan saat sidang daripada menyatakannya terus terang.

---

## Mencatat sesi

Pelabelan bekerja dengan mencocokkan **rentang waktu**. Jadi setiap sesi harus
tercatat waktu mulai dan selesainya dalam epoch.

Gunakan helper agar tidak salah konversi:

```bash
python -m ml_training.record_session --node NODE_A --label noise \
    --scenario "membanting pintu"
# Tekan ENTER untuk mulai, ENTER lagi untuk selesai.
# Sesi otomatis ditambahkan ke ml_training/sessions.csv
```

Isi `sessions.csv`:

| kolom | arti |
|---|---|
| `node_id` | node yang direkam |
| `label` | `earthquake` / `noise` / `gas_leak` |
| `scenario` | keterangan bebas, mis. "melompat" |
| `start_ts` | epoch mulai |
| `end_ts` | epoch selesai |
| `notes` | catatan kondisi, mis. "hujan, AC menyala" |

Berkas ini adalah **satu-satunya sumber kebenaran label**. Jangan melabeli
manual di CSV telemetri.

---

## Aturan kebersihan data

1. **Beri jeda 5 detik** di awal dan akhir tiap sesi sebelum menekan ENTER,
   supaya getaran dari tangan yang menekan tombol tidak ikut terlabeli.
2. **Satu sesi = satu skenario.** Jangan mencampur berjalan dan melompat dalam
   satu rentang waktu.
3. **Jangan merekam dua skenario bersamaan pada node yang sama.** Rentang waktu
   yang bertindihan akan ditolak oleh `label_dataset.py`.
4. **Catat kondisi tidak biasa** di `notes` — konstruksi di sebelah, angin
   kencang, node baru dipindah. Ini yang menyelamatkan saat metriknya aneh.
5. **Sebar sesi ke beberapa hari dan beberapa node.** Semua data dari satu sore
   di satu node akan menghasilkan model yang hanya mengenali sore itu.

---

## Target akhir sebelum lanjut ke Fase 2

- [ ] ≥ 6 skenario `noise`, masing-masing ≥ 30 jendela
- [ ] ≥ 100 jendela `earthquake`
- [ ] Data berasal dari ≥ 2 node berbeda
- [ ] Data tersebar di ≥ 3 hari berbeda
- [ ] `sessions.csv` lengkap dan tidak ada rentang bertindihan

Verifikasi jumlah jendela yang benar-benar terpicu:

```bash
python -m ml_training.export_dataset --out data/raw/cek.csv
python -m ml_training.label_dataset --telemetry data/raw/cek.csv --summary
```
