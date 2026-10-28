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

# Two data layers. Build hosts cache every layer, so each needs something that
# changes when it should re-run.
#
# 1. The full history (1,200+ sessions, minutes to download). Re-runs when the
#    archive's index changes. The index sits behind a CDN that can serve a copy up
#    to a day old, so this layer may lag; layer 2 catches up.
ADD https://mtf.trading/mcx/index.json /tmp/mtf_index.json
RUN python scripts/fetch_mtf.py --delay 0.05

# 2. What is new since layer 1, and the snapshot. Re-runs on EVERY build: the
#    Cloudflare trace page is different on each request, so a deploy-hook rebuild
#    always picks up the latest session and today's margin rate. Only missing
#    sessions are downloaded, so this takes seconds plus the engine run.
#    --require-latest fails the build if the newest session the index lists did
#    not arrive (the mirror mid-update), so the previous deploy keeps serving and
#    the nightly job's live-site check rebuilds it later.
ADD https://www.cloudflare.com/cdn-cgi/trace /tmp/build_time
RUN python scripts/fetch_mtf.py --delay 0.05 --require-latest \
 && python scripts/nightly.py --source csv --csv-dir data/raw --no-notify \
 && rm -rf data/processed/history

ENV MCX_SNAPSHOT_PATH=/app/data/processed/snapshot.json
EXPOSE 8000
# Render sets PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
