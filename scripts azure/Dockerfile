FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Librerías nativas requeridas por MXNet/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libquadmath0 libgomp1 libstdc++6 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Versiones fijas (NumPy < 2.0) para compatibilidad con MXNet 1.9.1
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Tu app
COPY app.py /app/app.py

EXPOSE 5000
ENV PORT=5000

CMD ["python", "app.py"]


