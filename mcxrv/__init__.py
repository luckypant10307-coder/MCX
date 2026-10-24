"""
MCX Gold Cross-Contract Relative Value, organised by pipeline stage.

  stage1_intake            INTAKE & CHECKS      parse, flag, specs, lifecycle, store
  stage2_normalize_signal  MODEL & SIGNAL       normalize, carry fit, premium, residual, gates
  stage3_gates_decision    GATES & DECISION     costs, margin, risk board
  stage4_simulate_deliver  SIMULATE & EVALUATE  replay, ledger, attribution, event study

The pre-restructure module names (`ingest`, `carry`, `signal`, `backtest`, ...) still
work, both as attributes (`from mcxrv import carry`) and as dotted imports
(`from mcxrv.signal import SignalConfig`), so scripts, tests, the API and the nightly
job did not have to change.

`canonical` is the Postgres layer and needs psycopg. It is loaded on first use rather
than here, so the analysis runs on a machine with no database driver installed --
the same arrangement as before the restructure, when it lived only under `mcxrv.db`.
"""
from __future__ import annotations

import importlib
import sys

# ---- Stage 1: Intake & checks ----
from .stage1_intake import specs
from .stage1_intake import calendar
from .stage1_intake import validation      # was ingest
from .stage1_intake import anomaly
from .stage1_intake import config

# ---- Stage 2: Model & signal ----
from .stage2_normalize_signal import normalize
from .stage2_normalize_signal import half_life   # was carry
from .stage2_normalize_signal import z_score     # was signal
from .stage2_normalize_signal import pair_match  # was liquidity

# ---- Stage 3: Gates & decision ----
from .stage3_gates_decision import liquidity_cost  # was costs
from .stage3_gates_decision import margin_capital  # was margin
from .stage3_gates_decision import risk
from .stage3_gates_decision import spanfile

# ---- Stage 4: Simulate & evaluate ----
from .stage4_simulate_deliver import replay_driver    # was backtest
from .stage4_simulate_deliver import position_ledger  # was ledger
from .stage4_simulate_deliver import evaluation       # was attribution
from .stage4_simulate_deliver import eventstudy

# Old name -> module. Registered in sys.modules as well as bound here, because an
# attribute alone does not satisfy `import mcxrv.signal` or `from mcxrv.signal import X`.
_ALIASES = {
    "ingest": validation,
    "carry": half_life,
    "signal": z_score,
    "liquidity": pair_match,
    "costs": liquidity_cost,
    "margin": margin_capital,
    "backtest": replay_driver,
    "ledger": position_ledger,
    "attribution": evaluation,
}
for _old, _mod in _ALIASES.items():
    globals()[_old] = _mod
    sys.modules[f"{__name__}.{_old}"] = _mod
for _name in ("specs", "calendar", "anomaly", "normalize", "risk", "spanfile", "eventstudy"):
    sys.modules[f"{__name__}.{_name}"] = globals()[_name]
del _old, _mod, _name

_LAZY = {"canonical": ".stage1_intake.canonical"}


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "specs", "calendar", "validation", "canonical", "anomaly", "config",
    "normalize", "half_life", "z_score", "pair_match",
    "liquidity_cost", "margin_capital", "risk", "spanfile",
    "replay_driver", "position_ledger", "evaluation", "eventstudy",
    *_ALIASES,
]
