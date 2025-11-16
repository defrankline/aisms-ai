# =========
# 1) Base —
# =========
FROM python:3.12

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose flask port
EXPOSE 5001

# ✅ PRODUCTION SERVER: Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
