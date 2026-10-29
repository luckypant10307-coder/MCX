# One container: nightly job builds the snapshot at start-up, the API serves it and
# the dashboard. For production, run scripts/nightly.py on a schedule instead.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY . .
ENV MCX_SNAPSHOT_PATH=/app/data/processed/snapshot.json PORT=8000
EXPOSE 8000
CMD python scripts/nightly.py --no-notify && uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
