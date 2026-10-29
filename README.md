# MCX Gold Spread Monitor

MCX lists four gold futures that are, underneath, the same metal: GOLDM (100 g),
GOLDTEN (10 g), GOLDGUINEA (8 g coin) and GOLDPETAL (1 g). Their prices should move
together once you put them on the same footing. This project checks every evening
whether any pair has drifted apart more than usual, and tests whether trading those
gaps would actually have made money.

**Short answer: the gaps are real and they do close, but not by enough to pay for
trading them.** Over 146 simulated trades on real MCX data (Jan 2022 – Oct 2026),
the average trade made +17.5 bp before costs and **−5.3 bp after costs**.

Live dashboard: <https://mcx-iefc.onrender.com> (sign-in required).

**For reviewers:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the tech stack, the
full architecture, the engine stages, the nightly workflow, the data model and the
notification and assistant flows, as diagrams with explanations.

> Built with AI assistance (Claude). The design decisions, checks and conclusions
> are described below and in [docs/DETAILS.md](docs/DETAILS.md).

---

## How it works

1. **Data.** Daily MCX settlement prices (bhavcopy) for all four contracts, from
   [mtf.trading](https://mtf.trading). 1,236 sessions so far.
2. **Same footing.** Each price is converted to rupees per 10 g of fine gold. MCX
   quotes these contracts on different bases (GOLDM per 10 g, GOLDPETAL per 1 g) and
   at different purities (995 vs 999), so comparing raw prices is meaningless.
3. **Remove the normal gap.** Contracts expiring later trade higher (cost of
   carry). A daily fit across all expiries removes that, leaving a residual for each
   contract.
4. **Signal.** For each of the six pairs, the gap between residuals is turned into a
   z-score against its own recent history. Beyond ±2 is "unusual".
5. **Gates.** An unusual pair only counts as a trade if there is enough time to
   expiry, enough trading volume, and enough margin capital.
6. **Simulation.** A walk-forward backtest trades every signal at the *next* day's
   price (the settlement price is only known after the close), pays costs, and
   tracks the margin MCX would block each day.

## Results (real data)

| | |
|---|---|
| Trades | 146 |
| Average per trade, before costs | +17.5 bp |
| Average per trade, after costs | **−5.3 bp** |
| Winning trades, after costs | 47 % |
| Costs as share of gross profit | 152 % |
| Peak margin blocked | ₹8.98 lakh |
| Return on margin, per year | −2.3 % |

The edge needs costs below about 1.1 bp per side to break even; realistic MCX costs
are several times that.

## Margin: what is published and what is assumed

Margin decides how much capital a spread ties up. Two parts:

* **Outright margin rate: published.** Brokers republish MCXCCL's daily rates. The
  project stores 674 daily sheets (Jun 2023 – Mar 2026, Aditya Birla Money archive)
  plus today's rate from Zerodha's margin page, and uses them on the sessions they
  cover. Gold margin ranged from 7.25 % to about **28 % in February 2026**.
* **Spread credit: assumed (70 %).** How much MCX reduces margin for an offsetting
  pair is only in the MCXCCL SPAN file or a broker's margin calculator. It stays an
  assumption until measured.

In practice MCX's 6 % minimum margin binds for most spreads, so the spread credit
rarely changes the result.

## What the app does

* **Nightly job** (GitHub Actions, weekdays): downloads new sessions, reads today's
  margin rate, rebuilds the snapshot, and redeploys the site when a new session
  arrives.
* **Dashboard** (React): pair status (ALERT / NO_TRADE / QUIET), pair history,
  decision log, backtest, research tables and a data audit.
* **Alerts**: phone push (ntfy, web push) and an in-app notification centre when a
  pair becomes or stops being an ALERT, when margin rates change, or when a new MCX
  circular appears. Accounts and preferences are on Supabase.
* **Risk checks**: account drawdown kill switch, and an anomaly review queue for
  suspicious price rows.

* **Ask**: a chat page where signed-in users ask questions ("why is this pair
  NO_TRADE?", "what is the margin now?"). Gemini answers through read-only tools over
  the same snapshot the dashboard shows; it explains, never decides or advises. Needs
  `GEMINI_API_KEY`; limited to 40 questions per user per day.
* **Timestamped record**: each session's statuses and decisions are saved as a small
  JSON file and timestamped in Bitcoin through
  [OpenTimestamps](https://opentimestamps.org), on the `attestations` branch. Anyone
  can check that an alert existed on its date and was not edited afterwards:
  `python scripts/attest.py --verify attestations/<file>.json`. It proves timing,
  not that the signal was right.

**It never places orders.** It reads public data and sends notifications; nothing is
connected to a broker or bank account.

## Running it

```bash
pip install -r requirements-api.txt

python scripts/fetch_mtf.py                                  # download bhavcopy
python scripts/fetch_margins.py                              # today's margin rates
python scripts/nightly.py --source csv --csv-dir data/raw --no-notify
uvicorn api.main:app --reload                                # API on :8000

cd frontend && npm ci && npm run dev                         # dashboard on :5173
python -m pytest                                             # tests
```

Secrets (`SUPABASE_*`, `VAPID_*`, `NTFY_TOPIC`, `TINYFISH_API_KEY`) go in `.env`,
which is never committed. See `.env.example`.

## Layout

```
mcxrv/
  stage1_intake/            contract specs, trading calendar, validation, anomalies
  stage2_normalize_signal/  per-gram normalisation, carry fit, z-score, liquidity
  stage3_gates_decision/    costs, SPAN margin, risk board, MCXCCL file parser
  stage4_simulate_deliver/  backtest, position ledger, attribution, event study
  pipeline.py               the whole run as one call
  snapshot.py               the JSON the API serves
  published_margins.py      broker margin sheets
  notify.py, user_notify.py alerts
api/                        FastAPI
frontend/                   React + TypeScript
scripts/                    nightly job, downloaders, analysis report
tests/                      pytest
```

## Limitations

* GOLDTEN has data only from March 2025.
* Spread credit is assumed, not measured.
* Margin sheets are missing for 2022 and for March–October 2026; those sessions use
  a volatility-based estimate.
* Costs are modelled, not taken from real fills.

More detail on every step (validation, look-ahead checks, the margin model, the
database layer) is in [docs/DETAILS.md](docs/DETAILS.md).
