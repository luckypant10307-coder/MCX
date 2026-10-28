# One image that serves the React dashboard and the API, with REAL MCX data baked in.

# ---- stage 1: build the React app (frontend/) into static files ---------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: Python API + data ------------------------------------------------------
# The data is downloaded and the snapshot built at BUILD time, not at start-up:
# free hosts (Render) wipe the disk and restart the container whenever it idles, and
# re-downloading 1,200+ sessions on every cold start would take minutes. Baked in,
# the container starts in seconds and a failed download fails the BUILD, so the
# previous good deploy keeps serving instead of an empty one.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .
COPY --from=web /web/dist ./frontend/dist

# Cache-buster for the data layer. Build hosts cache every layer, so without this an
# unchanged Dockerfile would keep serving the first day's data forever. ADD of a URL
# is checksummed: the layers below re-run exactly when mtf.trading publishes a new
# session (its index changes), and stay cached when nothing new exists.
ADD https://mtf.trading/mcx/index.json /tmp/mtf_index.json

# Download every session (MCX bhavcopy via mtf.trading), then build the snapshot
# over the full history up to the latest session.
RUN python scripts/fetch_mtf.py --delay 0.05 --require-latest \
 && python scripts/nightly.py --source csv --csv-dir data/raw --no-notify \
 && rm -rf data/processed/history

ENV MCX_SNAPSHOT_PATH=/app/data/processed/snapshot.json
EXPOSE 8000
# Render sets PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
