# =========
# Builder stage: build wheels untuk semua dependencies (termasuk yg butuh compile)
# =========
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Paket build untuk lib yang butuh header (lxml, cryptography, Pillow, dll)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ \
    libxml2-dev libxslt1-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Siapkan folder kerja
WORKDIR /build

# Salin requirements ke layer terpisah demi caching
COPY requirements.txt /build/requirements.txt

# (Opsional tapi cepat) pastikan pip terbaru
RUN python -m pip install --upgrade pip

# Build wheels di folder /wheels agar bisa dipakai ulang di stage runtime
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# =========
# Runtime stage: hanya runtime libs + install dari wheels
# =========
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=1 \
    PORT=8000

# Install runtime libraries (tanpa toolchain) untuk dependen C
# - libxml2/libxslt: lxml
# - libffi/openssl: cryptography
# - libjpeg/zlib: Pillow
# - libmagic1: python-magic
# - tesseract-ocr: pytesseract (wrapper butuh binary tesseract)
# - graphviz: pydot/nipype bisa memanggil 'dot'
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 \
    libffi8 \
    openssl \
    libjpeg62-turbo zlib1g \
    libmagic1 \
    tesseract-ocr \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Buat user non-root
RUN useradd -m appuser
WORKDIR /app

# Salin requirements dan wheels dari builder
COPY --from=builder /build/requirements.txt /app/requirements.txt
COPY --from=builder /wheels /wheels

# Install semua paket dari wheels (tanpa internet / compile ulang)
RUN python -m pip install --no-index --find-links=/wheels -r /app/requirements.txt

# Salin source code aplikasi ke image
COPY . /app

# Ubah kepemilikan dan turun hak akses
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Ubah "app.main:app" jika modul/objek FastAPI kamu berbeda
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
