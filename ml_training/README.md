# Pipeline Pelatihan Model (Assignment 3)

Seluruh alur dari data mentah di PostgreSQL sampai artefak model yang dimuat
server, dalam urutan yang bisa diulang oleh orang lain.

Semua skrip di folder ini bersifat **offline**. Runtime server tidak pernah
mengimpornya dan tidak butuh scikit-learn.

---

## Prasyarat

```bash
pip install -r requirements.txt          # dependensi runtime server
pip install scikit-learn joblib numpy    # tambahan khusus pelatihan
```

Dua hal yang harus benar sebelum mulai — keduanya membuat seluruh data
terbuang bila dilewatkan:

1. **Ingester versi baru sudah ter-deploy**, sehingga `freq_hz` tersimpan:

   ```bash
   cd ../../prototype/grafana-stack && docker-compose up -d --build
   ```

   ```sql
   SELECT COUNT(*) FROM sensor_telemetry WHERE freq_hz IS NOT NULL;
   ```

   Angkanya harus bertambah seiring waktu. Kalau tetap nol, jangan lanjut.

2. **Sesi perekaman sudah dicatat** di `sessions.csv` — lihat
   [`RECORDING_PROTOCOL.md`](RECORDING_PROTOCOL.md).

---

## Alur lengkap

```
  PostgreSQL
      │  export_dataset.py      buang baris tanpa freq_hz
      ▼
  data/raw/telemetry.csv
      │  label_dataset.py       cocokkan dengan rentang waktu sessions.csv
      ▼
  data/raw/labeled.csv
      │  build_dataset.py       jendela → fitur, augmentasi, split per kejadian
      ▼
  data/processed/dataset_v1.csv  +  dataset_v1.meta.json
      │
      ├─ train_classifier.py    → ml/models/classifier.joblib + model_meta.json
      ├─ train_magnitude.py     → ml/models/magnitude.joblib + magnitude_meta.json
      └─ evaluate_baseline.py   → reports/baseline.json
```

### 1. Ekspor

```bash
python -m ml_training.export_dataset --out data/raw/telemetry.csv
```

Baris tanpa `freq_hz` dibuang di sini — frekuensi dominan adalah fitur wajib
dan tidak bisa diimputasi. Skrip mencetak berapa baris yang terbuang.

### 2. Pelabelan

```bash
python -m ml_training.label_dataset \
    --telemetry data/raw/telemetry.csv --out data/raw/labeled.csv

# lihat ringkasannya saja, tanpa menulis berkas
python -m ml_training.label_dataset --telemetry data/raw/telemetry.csv --summary
```

Yang perlu diperhatikan pada keluarannya adalah **jumlah jendela LAYAK**, bukan
jumlah baris. Rekaman panjang yang getarannya terlalu pelan menghasilkan banyak
baris tetapi nol data yang terpakai.

### 3. Bangun dataset

```bash
python -m ml_training.build_dataset \
    --labeled data/raw/labeled.csv --out data/processed/dataset_v1.csv
```

Split dilakukan per `session_id`, bukan per baris. Augmentasi hanya pada train.
Skrip memperingatkan bila dataset masih terlalu tipis.

### 4. Latih

```bash
python -m ml_training.train_classifier --dataset data/processed/dataset_v1.csv
python -m ml_training.train_magnitude  --labeled data/raw/labeled.csv
```

### 5. Evaluasi pembanding

```bash
python -m ml_training.evaluate_baseline \
    --dataset data/processed/dataset_v1.csv --out reports/baseline.json
```

Angka inilah yang disandingkan dengan `test_report` di `model_meta.json` untuk
menjawab butir (h) requirement soal.

---

## Reproducibility

Seluruh skrip memakai `--seed` dengan nilai tetap 42. Seed, rasio split, daftar
fitur, dan distribusi kelas per split tercatat di `dataset_v1.meta.json`;
metrik, versi model, dan baseline distribusi fitur tercatat di
`model_meta.json`.

Menjalankan ulang seluruh rangkaian pada CSV yang sama akan menghasilkan model
dan metrik yang sama persis. Bila tidak, ada yang salah — laporkan, jangan
diulang sampai angkanya terlihat bagus.

---

## Melatih ulang saat data bertambah

Ulangi langkah 1–5. Tidak ada state tersembunyi di antara langkah; semuanya
mengalir lewat berkas. Naikkan nama versi dataset (`dataset_v2.csv`) agar
perbandingan antar versi tetap bisa dilacak.

Indikator bahwa pelatihan ulang memang diperlukan datang dari
`ml/drift_monitor.py`, yang membandingkan distribusi fitur yang sedang masuk
terhadap baseline pelatihan. PSI di atas 0,25 pada beberapa fitur berarti
kondisi lapangan sudah berubah.

---

## Menyebarkan model

1. Salin `classifier.joblib` dan `model_meta.json` ke `ml/models/` di server.
2. Restart `consensus.py`. Log akan menyatakan versi model yang dimuat.
3. **Biarkan shadow mode aktif** (`ML_SHADOW_MODE=true`, yaitu default) selama
   beberapa hari. Prediksi dicatat penuh tetapi tidak menyentuh aktuator.
4. Tinjau panel "Rule-Based vs ML" di Grafana, khususnya baris ketika kedua
   jalur tidak sepakat.
5. Baru setelah itu pertimbangkan `ML_SHADOW_MODE=false`.

Server menolak memuat model yang urutan fiturnya tidak cocok dengan
`feature_extractor.py`, lalu berjalan dengan rule-based saja. Jadi model lama
yang tertinggal tidak akan diam-diam dipakai setelah daftar fitur berubah.

---

## Kegagalan yang sering terjadi

| Gejala | Penyebab |
|---|---|
| `0 baris punya freq_hz` | ingester versi baru belum ter-deploy |
| `Tidak ada jendela yang layak` | getaran saat merekam kurang kuat (< 0,12 G) |
| `split 'test' kosong` | terlalu sedikit sesi; tambah perekaman |
| `Hanya ada satu kelas` | belum merekam salah satu dari noise/earthquake |
| `Model TIDAK mengalahkan persistensi` | hasil yang sah — laporkan apa adanya |
| Server: `urutan fitur tidak cocok` | model dilatih sebelum `FEATURE_NAMES` berubah; latih ulang |
