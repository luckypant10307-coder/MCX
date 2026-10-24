"""
FastAPI app: thin routing over api/service.py.

    uvicorn api.main:app --reload            # http://localhost:8000/docs

The dashboard in web/ is served at / so one container hosts both. On Render, set
MCX_SNAPSHOT_PATH if the snapshot lives elsewhere, and CORS_ORIGINS to the Vercel
URL if the dashboard is deployed separately.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import service

store = service.SnapshotStore()
app = FastAPI(title="MCX Gold Spread Monitor API", version="0.1.0",
              description="Read-only API over the nightly snapshot. Every decision "
                          "carries the reason that produced it.")

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"],
                   allow_headers=["*"])


def _wrap(fn, *a, **kw):
    try:
        return fn(store, *a, **kw)
    except service.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/health")
def health():
    return service.health(store)


@app.get("/api/overview")
def overview():
    return _wrap(service.overview)


@app.get("/api/pairs")
def pairs(status: str | None = Query(None, description="ALERT, NO_TRADE or QUIET")):
    return _wrap(service.pairs, status=status)


@app.get("/api/pairs/{pair}")
def pair_detail(pair: str):
    """Pair as GOLDM-GOLDTEN (a dash, since a slash cannot sit in a path segment)."""
    return _wrap(service.pair_detail, pair)


@app.get("/api/decisions")
def decisions(limit: int = Query(50, ge=1, le=1000), status: str | None = None,
              pair: str | None = None):
    return _wrap(service.decisions, limit=limit, status=status, pair=pair)


@app.get("/api/risk")
def risk_board():
    return _wrap(service.risk_board)


@app.get("/api/data-quality")
def data_quality():
    return _wrap(service.data_quality)


@app.get("/api/backtest")
def backtest():
    return _wrap(service.backtest)


@app.get("/api/research")
def research():
    """Full-history evidence: event study, cost sweep, premiums, coverage, by-year."""
    return _wrap(service.research)


WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))
