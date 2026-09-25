FROM python:3.10-slim

# Set timezone (opsional tapi bagus untuk log server)
ENV TZ=Asia/Jakarta

WORKDIR /app

# Install dependencies terlebih dahulu (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode program
COPY consensus.py .

# Paket lapisan kecerdasan. Tanpa baris ini consensus.py tetap berjalan,
# tetapi init_ml() gagal mengimpor dan sistem diam-diam kembali ke rule-based
# saja — persis kegagalan senyap yang paling sulit disadari, karena tidak ada
# error yang muncul dan deteksi gempa tetap bekerja.
COPY ml/ ./ml/

# Artefak model TIDAK disalin ke dalam image. Model berubah jauh lebih sering
# daripada kode, dan menanamnya di image berarti harus membangun ulang seluruh
# image setiap kali model dilatih ulang. Pasang lewat volume:
#   volumes:
#     - ./src/server/ml/models:/app/ml/models:ro
# Bila direktori itu kosong, server berjalan dengan rule-based saja.

# Jalankan server secara terus-menerus
CMD ["python", "-u", "consensus.py"]
