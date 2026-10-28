"""
"Ask" assistant: questions about the monitor, answered from the nightly snapshot.

Gemini (Google Gen AI SDK) is given read-only tools over the same snapshot the
dashboard shows, and is told to answer only from what they return. It cannot change
anything, and it explains what the system says rather than advising a trade.

Configuration (environment):
  GEMINI_API_KEY      required. A Vertex AI express-mode key ("AQ...") or an AI
                      Studio key ("AIza..."); the prefix picks the endpoint.
  GEMINI_MODEL        default gemini-2.5-flash
  CHAT_DAILY_LIMIT    questions per user per UTC day, default 40
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_QUESTION_CHARS = 1000
MAX_HISTORY_TURNS = 10

SYSTEM = """You are the assistant inside the MCX Gold Spread Monitor, a research tool that
compares four MCX gold futures (GOLDM 100 g, GOLDTEN 10 g, GOLDGUINEA 8 g, GOLDPETAL 1 g)
after converting each to rupees per 10 g of fine gold and removing normal carry between
expiries. Each of six pairs gets a z-score; beyond +/-2 is unusual. An unusual pair is an
ALERT only if every gate passes (time to expiry, liquidity, costs, margin, risk board,
kill switch); otherwise NO_TRADE with the failed gate named. QUIET means inside +/-2.
WARMUP means not enough history yet.

Rules:
- Answer only from the tools. Call them for every number, status or date you mention;
  never estimate or recall figures. If the tools do not have it, say so.
- Name the session date ("as of ...") the answer comes from.
- Explain; do not advise. Never tell the user to buy, sell or size a position. If asked,
  say what the system shows and remind them the backtest found no edge after costs.
- Margin rates are published broker sheets on most sessions; the spread credit is an
  assumption. Say which when it matters.
- Be short: a few sentences or a small list. Plain words; explain jargon on first use.
- Reply in the language the user writes in (English, Hindi or Hinglish)."""


class ChatUnavailable(RuntimeError):
    """The assistant is not configured or the model could not be reached."""


class ChatLimit(RuntimeError):
    """This user has used today's questions."""


# ------------------------------------------------------------------ tools

def tools_for(snap: dict) -> list:
    """Read-only functions over one snapshot, in the form the SDK can call."""
    meta = snap.get("meta", {})

    def get_overview() -> dict:
        """Session date, data source, how many pairs are in each status, the backtest
        verdict, and the margin parameters in use."""
        return {"asof": meta.get("asof"), "sessions": meta.get("sessions"),
                "source": meta.get("source"), "sample_data": meta.get("sample_data"),
                "status_counts": snap.get("summary"),
                "verdict": (snap.get("backtest") or {}).get("verdict"),
                "margin_parameters": meta.get("span_parameters")}

    def get_pairs() -> list:
        """Latest status, z-score, reason and failed gate for all six pairs."""
        return [{k: p.get(k) for k in ("pair", "status", "z", "reason", "binding_gate",
                                       "spread_bp", "dislocation_bp", "halflife")}
                for p in snap.get("pairs", [])]

    def get_pair(pair: str) -> dict:
        """Full detail for one pair, written like GOLDM/GOLDTEN: gates, risk board row,
        legs, and the last 30 sessions of z-scores."""
        want = pair.upper().replace("-", "/").replace(" ", "")
        for p in snap.get("pairs", []):
            if p["pair"].replace(" ", "") == want or set(p["pair"].split("/")) == set(want.split("/")):
                hist = (snap.get("pair_history") or {}).get(p["pair"], [])
                return {**{k: v for k, v in p.items() if k != "telegram"},
                        "recent_z": hist[-30:] if isinstance(hist, list) else hist}
        return {"error": "unknown pair {!r}; pairs are {}".format(
            pair, ", ".join(p["pair"] for p in snap.get("pairs", [])))}

    def get_decisions(limit: int = 20, pair: str = "") -> list:
        """Recent unusual sessions, newest first: TRADED, SIGNAL or NO_TRADE with the
        reason. Optionally only one pair (like GOLDM/GOLDTEN)."""
        rows = list(snap.get("decision_log", []))          # stored newest first
        if pair:
            want = set(pair.upper().replace("-", "/").split("/"))
            rows = [r for r in rows if set(r.get("pair", "").split("/")) == want]
        return rows[:max(1, min(int(limit), 60))]

    def get_backtest() -> dict:
        """Backtest performance on real data: trades, gross and net bp per trade, hit
        rates, peak margin, and results by year and by pair."""
        r = snap.get("research") or {}
        return {"performance": (snap.get("backtest") or {}).get("performance"),
                "verdict": (snap.get("backtest") or {}).get("verdict"),
                "by_year": r.get("by_year"), "by_pair": r.get("by_pair"),
                "exit_reasons": r.get("exit_reasons"), "window": r.get("window")}

    def get_account() -> dict:
        """Simulated account against its kill switch: value, peak, drawdown, halted."""
        return snap.get("account") or {}

    def get_margins() -> dict:
        """Published outright margin rates per contract: latest, recent changes, and
        the dates covered."""
        m = dict(snap.get("published_margins") or {"available": False})
        hist = m.pop("history", None) or []
        if hist:
            m["recent_history"] = hist[-40:]
        return m

    def get_circulars() -> dict:
        """MCX circulars found by web search, newest first, with which are new."""
        c = dict(snap.get("circulars") or {"available": False})
        c["items"] = (c.get("items") or [])[:10]
        return c

    def get_data_quality() -> dict:
        """Rows in and out, validation flags, and the anomaly review summary."""
        an = snap.get("anomalies") or {}
        return {"data_quality": snap.get("data_quality"),
                "anomalies": {k: an.get(k) for k in an if k not in ("rows", "queue")}}

    return [get_overview, get_pairs, get_pair, get_decisions, get_backtest,
            get_account, get_margins, get_circulars, get_data_quality]


# ------------------------------------------------------------------ limits

_USE: dict = {}
_LOCK = threading.Lock()


def take_question(user_id: str, limit: int | None = None) -> int:
    """Count one question for today; returns how many are left. Raises ChatLimit."""
    limit = int(os.environ.get("CHAT_DAILY_LIMIT", "40")) if limit is None else limit
    day = datetime.now(timezone.utc).date().isoformat()
    with _LOCK:
        for k in [k for k in _USE if k[1] != day]:
            del _USE[k]
        used = _USE.get((user_id, day), 0)
        if used >= limit:
            raise ChatLimit("You have used today's {} questions. They reset at 05:30 IST.".format(limit))
        _USE[(user_id, day)] = used + 1
        return limit - used - 1


def give_back(user_id: str) -> None:
    """Return a question that failed on our side (model unreachable), not the user's."""
    day = datetime.now(timezone.utc).date().isoformat()
    with _LOCK:
        if _USE.get((user_id, day), 0) > 0:
            _USE[(user_id, day)] -= 1


# ------------------------------------------------------------------ model

def configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _client():
    from google import genai
    key = os.environ["GEMINI_API_KEY"]
    # Vertex AI express-mode keys start "AQ."; AI Studio keys start "AIza".
    return genai.Client(vertexai=not key.startswith("AIza"), api_key=key)


def answer(question: str, history: list | None, snap: dict, client=None) -> dict:
    """
    One reply. `history` is [{"role": "user"|"assistant", "text": ...}] from the page,
    trimmed to the last few turns. Returns {"text", "tools", "model", "asof"}.
    """
    from google.genai import types

    question = (question or "").strip()
    if not question:
        raise ValueError("Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError("Keep the question under {} characters.".format(MAX_QUESTION_CHARS))
    if not configured() and client is None:
        raise ChatUnavailable("The assistant is not set up on this server (GEMINI_API_KEY).")

    contents = []
    for turn in (history or [])[-2 * MAX_HISTORY_TURNS:]:
        text = str(turn.get("text", ""))[:4000]
        if text:
            contents.append(types.Content(role="model" if turn.get("role") == "assistant" else "user",
                                          parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    used: list = []
    tools = []
    for fn in tools_for(snap):
        tools.append(_recording(fn, used))
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=tools, temperature=0.2, max_output_tokens=900,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=6))
    c = client or _client()
    try:
        # A chat session is the SDK's supported home for automatic tool calling.
        chat = c.chats.create(model=model, config=cfg, history=contents[:-1])
        resp = chat.send_message(question)
    except Exception as e:
        raise ChatUnavailable(_explain(e)) from e
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        text = "I could not produce an answer to that. Try asking it another way."
    return {"text": text, "tools": used, "model": model,
            "asof": (snap.get("meta") or {}).get("asof")}


def _recording(fn, used: list):
    """Wrap a tool so the reply can say which data it read. Keeps name, docstring
    and signature, which the SDK reads to describe the tool to the model."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        used.append(fn.__name__)
        return fn(*a, **kw)
    return wrapper


def _explain(e: Exception) -> str:
    s = str(e)
    if "SERVICE_DISABLED" in s or "has not been used in project" in s:
        return ("The Gemini API is not enabled for this key's Google Cloud project. "
                "Enable the Vertex AI (Agent Platform) API in the Google Cloud console.")
    if "PERMISSION_DENIED" in s or "API_KEY_INVALID" in s or "401" in s:
        return "Google refused the API key. Check GEMINI_API_KEY."
    if "RESOURCE_EXHAUSTED" in s or "429" in s:
        return "The Gemini quota is used up for now. Try again in a minute."
    if "NOT_FOUND" in s or "404" in s:
        return "The model {} is not available for this key. Set GEMINI_MODEL.".format(
            os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    return "The assistant could not reach Gemini ({}).".format(type(e).__name__)
