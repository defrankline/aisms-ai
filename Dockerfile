# =========
# 1) Base —
# =========
FROM python:3.12-slim

# Prevent interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# =========
# 2) System deps
# =========
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# =========
# 3) Workdir
# =========
WORKDIR /app

# =========
# 4) Install Python deps
# =========
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# =========
# 5) Copy app
# =========
COPY . .

# =========
# 6) Expose port
# =========
EXPOSE 5001

# =========
# 7) Start
# =========
CMD ["python", "app.py"]