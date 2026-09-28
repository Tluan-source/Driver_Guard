# DriverGuard backend + built dashboard in one image.
#   docker compose up --build            (from the repo root)
# Build context = repo root (see docker-compose.yml).

# ---- stage 1: dashboard ----
FROM node:20-slim AS dashboard
WORKDIR /app/dashboard
COPY dashboard/package.json ./
RUN npm install --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# ---- stage 2: python runtime ----
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY driverguard/ driverguard/
COPY configs/ configs/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e ".[api]"
COPY --from=dashboard /app/dashboard/dist /app/dashboard/dist
# models/ and data/ are mounted as volumes (never baked into the image)
EXPOSE 8000
CMD ["driverguard", "serve", "--source", "synthetic:demo", "--host", "0.0.0.0", "--port", "8000"]
