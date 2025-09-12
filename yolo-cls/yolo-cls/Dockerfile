FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Libs runtime mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Crea usuario no root con HOME real
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Rutas de caché ESCRIBIBLES (Ultralytics/Torch/etc)
ENV HOME=/home/appuser \
    XDG_CACHE_HOME=/app/.cache \
    TORCH_HOME=/app/.cache/torch \
    ULTRALYTICS_CACHE_DIR=/app/.cache/ultralytics \
    MPLCONFIGDIR=/app/.cache/mpl

# Crea y da permisos a las carpetas
RUN mkdir -p /app/.cache && chown -R 1000:1000 /app /home/appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8000
USER 1000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

