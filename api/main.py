"""
FastAPI app: thin routing over api/service.py.

    uvicorn api.main:app --reload            # http://localhost:8000/docs

The dashboard in web/ is served at / so one container hosts both. On Render, set
MCX_SNAPSHOT_PATH if the snapshot lives elsewhere, and CORS_ORIGINS to the Vercel
URL if the dashboard is deployed separately.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api import service
from mcxrv import supa
from mcxrv.stage1_intake.config import load_dotenv

load_dotenv()                     # local .env; real environment variables win
store = service.SnapshotStore()
app = FastAPI(title="MCX Gold Spread Monitor API", version="0.2.0",
              description="Read-only API over the nightly snapshot. Every decision "
                          "carries the reason that produced it.")

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])

# Routes anyone may call. Everything else under /api/ needs a signed-in user whenever
# accounts are switched on (SUPABASE_URL + SUPABASE_PUBLISHABLE_KEY set).
PUBLIC_API = {"/api/health", "/api/config"}


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if (path.startswith("/api/") and path not in PUBLIC_API and request.method != "OPTIONS"
            and supa.public_config()["auth_enabled"]):
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        user = supa.user_from_token(token)
        if not user:
            return JSONResponse({"detail": "Sign in to see this."}, status_code=401)
        request.state.user = user
    return await call_next(request)


@app.get("/api/config")
def config():
    """Public settings the browser needs to sign in and enable push."""
    return supa.public_config()


@app.get("/api/me")
def me(request: Request):
    u = getattr(request.state, "user", None) or {}
    return {"id": u.get("id"), "email": u.get("email"),
            "provider": (u.get("app_metadata") or {}).get("provider")}


@app.post("/api/notifications/test")
def send_test(request: Request):
    """Send the signed-in user a test notification on every device they enabled."""
    u = getattr(request.state, "user", None)
    if not u:
        raise HTTPException(status_code=401, detail="Sign in first.")
    if not supa.configured():
        raise HTTPException(status_code=503, detail="Notifications are not configured on this server.")
    from mcxrv import user_notify
    res = user_notify.fanout([{"kind": "test", "title": "Test notification",
                               "body": "Notifications are working on this device.", "url": "/settings"}],
                             run_id="test", only_user=u["id"])
    return res


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


@app.get("/api/circulars")
def circulars():
    """MCX margin / specification circulars found by TinyFish search; new ones marked."""
    return _wrap(service.circulars)


@app.get("/api/risk-files")
def risk_files():
    """The MCXCCL drop folder: files accepted, rejected (with reasons) and coverage."""
    return _wrap(service.risk_files)


@app.get("/api/anomalies")
def anomalies():
    """Isolation Forest review queue: flagged rows for a human. Never gates data."""
    return _wrap(service.anomalies)


@app.get("/api/settings")
def settings():
    """Engine parameters used by this run, and per-pair z history for what-if counts."""
    return _wrap(service.settings)


@app.get("/api/research")
def research():
    """Full-history evidence: event study, cost sweep, premiums, coverage, by-year."""
    return _wrap(service.research)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The React app (frontend/, built by `npm run build`) when present; otherwise the
# single-file dashboard in web/, so the API still has a UI without a Node build.
REACT_DIST = os.path.join(ROOT, "frontend", "dist")
WEB_DIR = REACT_DIST if os.path.isfile(os.path.join(REACT_DIST, "index.html")) else os.path.join(ROOT, "web")

if os.path.isdir(WEB_DIR):
    INDEX = os.path.join(WEB_DIR, "index.html")
    if os.path.isdir(os.path.join(WEB_DIR, "assets")):
        # Hashed file names never change content, so they can be cached for a year.
        app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(INDEX, headers={"Cache-Control": "no-cache"})

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        """Client-side routes (/pairs/GOLDM-GOLDTEN, /research ...) load the app shell,
        so a refresh or a shared link opens the right page. Unknown /api paths stay 404."""
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        # Real files at the site root (service worker, manifest, icons) are served as
        # themselves. The service worker is never cached, so updates reach every device.
        candidate = os.path.normpath(os.path.join(WEB_DIR, path))
        if path and candidate.startswith(os.path.normpath(WEB_DIR) + os.sep) and os.path.isfile(candidate):
            headers = {"Cache-Control": "no-cache"} if path == "sw.js" else {}
            media = "application/manifest+json" if path.endswith(".webmanifest") else None
            return FileResponse(candidate, headers=headers, media_type=media)
        return FileResponse(INDEX, headers={"Cache-Control": "no-cache"})
