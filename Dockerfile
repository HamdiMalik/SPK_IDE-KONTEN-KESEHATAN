# Dockerfile
# Konfigurasi build image Docker untuk backend Flask SPK Konten Kesehatan.
# Base image: python:3.10-slim (ringan, stabil, kompatibel dengan PyTorch)

# -------------------------------------------------------------------
# STAGE: Build Image
# -------------------------------------------------------------------
FROM python:3.10-slim

# Metadata image
LABEL maintainer="Hamdi <hamdi@example.com>"
LABEL description="Backend Flask untuk SPK Pembuatan Ide Konten Kesehatan"
LABEL version="1.0.0"

# -------------------------------------------------------------------
# Konfigurasi Environment
# -------------------------------------------------------------------

# Mencegah Python membuat file .pyc (bytecode cache) — menghemat space
ENV PYTHONDONTWRITEBYTECODE=1

# Menonaktifkan buffer stdout/stderr Python — penting agar log langsung
# tertulis ke container log tanpa delay
ENV PYTHONUNBUFFERED=1

# Direktori kerja di dalam container
WORKDIR /app

# -------------------------------------------------------------------
# Instalasi Dependensi Sistem
# Paket yang dibutuhkan:
# - build-essential : Compiler C/C++ (gcc, make) untuk PyTorch dan psycopg2
# - libpq-dev       : Header library PostgreSQL untuk kompilasi psycopg2
# - curl            : Berguna untuk debugging health check di dalam container
# - git             : Dibutuhkan beberapa library Transformers untuk download config
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Instalasi Dependensi Python
# Copy requirements.txt terlebih dahulu (sebelum source code) untuk
# memanfaatkan Docker layer caching. Layer ini hanya di-rebuild jika
# requirements.txt berubah, bukan setiap kali ada perubahan kode.
# -------------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

# -------------------------------------------------------------------
# Copy Source Code Aplikasi
# Copy dilakukan setelah instalasi dependensi agar layer dependensi
# bisa di-cache Docker dan tidak perlu diinstall ulang setiap build.
# -------------------------------------------------------------------
COPY . .

# -------------------------------------------------------------------
# Buat direktori untuk model IndoBERTweet yang di-fine-tune.
# Model aktual akan di-mount melalui volume atau disalin saat build.
# -------------------------------------------------------------------
RUN mkdir -p /app/saved_models/indobert_sentiment

# -------------------------------------------------------------------
# Expose Port
# Memberitahu Docker bahwa container mendengarkan pada port 5000.
# Port ini harus sesuai dengan variabel PORT di .env
# -------------------------------------------------------------------
EXPOSE 5000

# -------------------------------------------------------------------
# Health Check
# Docker akan secara periodik memeriksa apakah Flask API berjalan normal.
# --interval : Selang waktu antar pemeriksaan
# --timeout  : Batas waktu tunggu response
# --retries  : Jumlah percobaan sebelum container dinyatakan unhealthy
# --start-period: Waktu tunggu awal sebelum health check pertama
#                 (penting untuk IndoBERTweet yang butuh waktu load)
# -------------------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --retries=5 \
    --start-period=60s \
    CMD curl -f http://localhost:5000/health || exit 1

# -------------------------------------------------------------------
# Perintah Startup
# Jalankan aplikasi Flask melalui app.py (bukan flask run, agar
# pengaturan host, port, dan debug dapat dikontrol dari kode)
# -------------------------------------------------------------------
CMD ["python", "app.py"]
