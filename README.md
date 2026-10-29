# MCX Gold Cross-Contract Relative Value

A data product that turns MCX gold settlement data into a defensible relative-value
signal — and then establishes, honestly, how much of that signal survives costs.

**Headline result on the validation data: the dislocation is real and statistically
strong, and it does not clear transaction costs with confidence.** Convergence runs to
t = 11.8 at |z| 2–3 over 20 sessions, yet **zero of 20 band–horizon cells clear the
cost line with a 95 % bootstrap interval**, and the walk-forward backtest turns
+10.7 bp gross into −35.0 bp net. That is reported as the finding rather than buried,
because the brief explicitly accepts a rigorous negative as a valid outcome.

---

## The thing most attempts get wrong

MCX quotes these four contracts on **different quotation bases**, which are not the
same as their trading units:

| Contract | Trading unit | **Price quoted per** | Purity | Expiry window | Deliverable |
|---|---|---|---|---|---|
| GOLDM | 100 g | **10 g** | 995 | 3rd–5th | bar |
| GOLDTEN | 10 g | **10 g** | 999 | 27th–31st | bar |
| GOLDGUINEA | 8 g | **8 g** | 999 | 27th–31st | coin |
| GOLDPETAL | 1 g | **1 g** | 999 | 27th–31st | coin |

Bhavcopy `Close` is the **quote**, never the contract notional. GOLDM at 12,000 is
₹12,000 per 10 g — the ₹1.2 lakh figure is the lot value and appears nowhere in the
file. So **GOLDM and GOLDTEN already share a quotation base** and differ only in
purity. Dividing GOLDM by its 100 g trading unit — the intuitive move — makes it
look 10× cheap and produces a spread series that is pure arithmetic error.

The single conversion, to rupees per 10 g of **fine** gold:

```python
norm_close = close * 10 / (quote_base_g * purity)
```

Purity alone puts GOLDM ~40 bp below the 999 contracts (0.999/0.995 = 1.00402),
mechanically, with no mispricing whatsoever.

## Three more traps, and what this repo does about them

**GOLDM never shares an expiry with the others.** It settles 3rd–5th; the 999 trio
settle 27th–31st. Every GOLDM cross-spread therefore contains ~23 days of carry.
Anchoring all spreads on GOLDM — the obvious choice — puts calendar basis into every
one of them. `GOLDTEN / GOLDGUINEA / GOLDPETAL` is the clean maturity-matched
triangle, and the dashboard labels GOLDM pairs "carry-bridged" so a reader can see
how much rests on the carry model.

**The coin contracts carry a permanent, justified premium.** GOLDPETAL and
GOLDGUINEA deliver minted coins; fabricating 1 g petals costs far more per gram than
casting a bar. That premium is *economically correct* and does not mean-revert. A
parity spread reads it as a standing opportunity and shorts it forever. Here it is
estimated as a structural term and removed before anything is called a signal.

**Nothing enforces convergence.** You cannot take delivery of a GOLDM bar and tender
it against GOLDGUINEA — the deliverable forms differ, so there is no arbitrage band
holding these contracts together, and each settles against its own reference price.
The relationship is statistical, not enforced. The repo states this rather than
assuming mean reversion, and tests stationarity and half-life before trading.

## The model

In log space, for contract *i* of symbol *s* on day *t*:

```
ln P_norm[i,t] = a_t  +  b_t · τ[i,t]  +  c_s  +  e[i,t]
```

| Term | Meaning | Actionable? |
|---|---|---|
| `a_t` | log level of fine gold | No — this is the gold factor |
| `b_t · τ` | carry slope × time to expiry | No — mechanical roll-down |
| `c_s` | structural premium (fabrication, form, participants) | No — persistent by design |
| `e[i,t]` | residual | **Yes — the only candidate signal** |

**Identification doubles as the no-look-ahead guarantee.** Within one day, `c_s` and a
symbol-level dislocation are the same object — both are a constant on every contract
of that symbol. They separate only through time, by persistence. So `c_s` is
estimated **only from a trailing window ending at t−1**, while `a_t` and `b_t` are fit
on day *t*'s cross-section alone. There is no code path by which a future observation
reaches day *t*'s residual. `GOLDM` is pinned to zero, so `a_t` is GOLDM's own level
and every premium is quoted against the 995 bar.

## Validation

Three things are established that real data alone cannot establish.

**1. The model recovers a truth it was never shown.** Against the synthetic
generator's ground truth:

| Quantity | True | Recovered |
|---|---|---|
| log level `a_t` | — | corr **0.9999** |
| carry slope `b_t` | 0.0587 mean | 0.0635 mean, corr 0.70 |
| GOLDGUINEA premium | 35.0 bp | **31.1 bp** |
| GOLDPETAL premium | 110.0 bp | **108.1 bp** |
| symbol dislocation | 22 bp sd | corr **0.89** (GOLDGUINEA) |

GOLDM's measured residual has sd **19.5 bp** despite a true residual of exactly zero.
That is the pure measurement-noise floor — tick quantisation plus curve-fit error —
and it is the *same order as the real signal*. Most of what a naive model would call
GOLDPETAL dislocation is noise. The signal gate compares against this floor.

**2. Mutating the future does not change the past.** `tests/test_no_lookahead.py` fits
the full sample, fits a truncated sample, and asserts that every residual, level,
slope, premium and z-score on shared dates is **bit-identical**. A look-ahead leak
cannot pass this.

**3. The null test — and it caught a real bug.** Set `residual_bp=0` and the world has
no relative value by construction. With same-bar entry the event study reported
**t-statistics above 30** on that data. The cause: signed convergence measured as
`-sign(z)·(spread[t+h] − spread[t])` contains `+|spread[t] − centre|`, which is
positive by construction, so *any* noise manufactures apparent mean reversion —
conditioning on a large deviation guarantees the next draw sits closer to the mean.
Entering one session later breaks the shared noise term. That is also the only honest
execution assumption: a settlement is published after the close, so a signal derived
from it cannot be filled at it.

The discriminating signature is worth keeping: **noise-driven pseudo-convergence is
flat in horizon**; a real mean-reverting signal **grows** with horizon toward its
half-life.

| |z| band, h=1→20 | Null world (no edge) | Real world |
|---|---|---|
| lag-0 (biased) | 3.81 → 3.56 bp, *flat* | — |
| lag-1 (correct) | −0.38 → −0.14 bp, ~zero | 1.46 → **21.88 bp**, growing |

## Results

Event study, all pair-sessions, entry lagged one session:

| \|z\| band | h | n | Mean conv. | Hit rate | t (Newey-West) | t (non-overlap) | Net of 33.1 bp cost |
|---|---|---|---|---|---|---|---|
| 0–1 | 20 | 3179 | +7.0 bp | 60 % | 7.3 | 3.9 | −26.1 |
| 1–2 | 10 | 1374 | +13.0 bp | 71 % | 10.2 | 7.7 | −20.1 |
| 1–2 | 20 | 1365 | +22.6 bp | 80 % | 12.3 | 5.0 | −10.5 |
| 2–3 | 10 | 264 | +22.6 bp | 83 % | 10.0 | 4.2 | −10.5 |
| 2–3 | 20 | 264 | +34.5 bp | 94 % | 11.8 | 5.3 | **+1.4** |
| 3+ | 20 | 25 | +47.4 bp | 100 % | 27.8 | — | **+14.3** |

Convergence is monotone in both signal strength and horizon — the signature of a real
effect. Newey-West cuts the naive t from 30.0 to 12.3, so the overlap correction
matters. **Of 20 band–horizon cells, 3 beat the cost line on the point estimate and
zero clear it with a 95 % bootstrap interval.**

Walk-forward backtest (executable subset):

| | |
|---|---|
| Trades | 11 |
| Gross per trade | **+10.7 bp** |
| Net per trade | **−35.0 bp** |
| Net hit rate | 9 % (gross 64 %) |
| Gold beta | −0.0003, 95 % CI [−0.0010, +0.0003] |
| Mean unhedged gold | 1.2 % of fine grams |
| Peak margin blocked | ₹78,419 (6 % floor binds) |
| **Return on margin, annualised** | **−3.09 %** |
| Dominant skip reason | `skip_cannot_size` — lot granularity |

P&L attribution confirms it is relative value, not something else: the residual
convergence term carries ~68 % of gross movement, gold leak is under 1 bp, and
**roll-down is exactly zero** — both legs are entered and exited on the same sessions,
so they age identically and `b·(Δτ_a − Δτ_b)` cancels. What survives is the curve
term, which is a term-structure bet, not roll-down.

The binding constraints, in order: **transaction costs**, then **lot granularity**
(GOLDM:GOLDGUINEA is 24875/1998 in fine grams, so the smallest exactly neutral
position is 1998 × 24875 lots — about ₹24 crore a leg; exact neutrality is
unattainable and a residual always exists), then **runway** (only the front month is
liquid and it lives ~21 sessions, against a ~12-session half-life needing two
half-lives to converge).

## Running it

```bash
pip install -r requirements.txt

python scripts/run_analysis.py                    # full pipeline + written verdict
python scripts/run_analysis.py --null             # the falsification test
python scripts/make_dashboard.py                  # -> reports/dashboard.html
python -m pytest                                  # 199 tests
```

Useful flags: `--z-entry`, `--half-spread-bp` (override the cost assumption),
`--maturity-matched-only` (drop every carry-bridged pair), `--seed`.

Figures quoted in this README come from the Postgres store with the robust (Huber)
carry fit enabled — reproduce them with `--source db` after a `db_load.py ... --model`.

## PostgreSQL store

```bash
cp .env.example .env            # edit credentials; .env is gitignored
python scripts/db_init.py       # create database + schema
python scripts/db_init.py --check

python scripts/db_load.py --source synthetic --start 2021-01-01 --end 2026-09-18 --model
python scripts/db_load.py --source csv --csv-dir data/raw --model --backtest
python scripts/db_load.py --source live --since-last --model   # daily incremental
python scripts/db_load.py --model-only                         # refit, no refetch

python scripts/run_analysis.py --source db      # read validated quotes
python scripts/run_analysis.py --source db-raw  # replay raw through validation
python scripts/make_dashboard.py --source db
```

Connection settings come from `.env` or the environment (`MCX_DB_HOST`, `_PORT`,
`_NAME`, `_USER`, `_PASSWORD`, `_SCHEMA`). Explicit environment variables win over
`.env`, so a deployment setting cannot be overridden by a stray file. No credential
appears in source.

### The layering rule

**`bhavcopy_raw` is append-only and immutable. Everything else is derived and
rebuildable from it.**

Validation rules and model parameters *will* change. When they do, history has to be
re-derivable without re-fetching — so the raw tape stores exactly what the upstream
returned, mistakes included, and is never deduplicated. A duplicate in the upstream
response is part of what happened. Deduplication is a property of `quote`, not of the
tape.

`tests/test_db.py::test_raw_tape_is_sufficient_to_rebuild` enforces this: it replays
the stored tape through the full validation path and asserts the result matches what
was stored. If the tape were missing something, that test fails.

The load-bearing column is **`bhavcopy_raw.requested_date`** — the date *we asked
for*, kept beside the date MCX echoed back. Store only the echoed date and the
substitution becomes undetectable forever after. `v_echo_audit` reads straight off it:

```
requested_date   raw_date  rows
    2021-01-02 01/01/2021    32     <- Saturday requested, Friday returned
    2021-01-09 01/08/2021    32
```

### Idempotency

Every derived table upserts on its natural key, so reloading a range converges
instead of accumulating. Verified by test and observable directly:

| Table | After 1st load | After identical 2nd load |
|---|---|---|
| `bhavcopy_raw` | 40,555 | **81,110** — append-only, by design |
| `quote` | 27,345 | **27,345** |
| `contract` | 242 | **242** |
| `pair_signal` | 5,499 | **5,499** |

`quote.run_id` is rewritten on upsert, so the data-quality ledger can be attributed
to the runs that produced the *current* rows — otherwise a reloaded range would
double-count its own flags.

### Specification versioning

Contract specs get revised, and a single static spec per symbol means that the moment
one is, the new value is applied silently to *all of history* — rewriting every past
normalised price with no error raised. Two defences:

**1. Specs are effective-dated by the contract's LISTING date.** A revision applies to
contracts listed *after* it, never retroactively to contracts already trading — which
is how exchanges actually work. So two expiries of the same symbol can be governed by
different specs on the same session, and `specs.get(symbol, on=listing_date)` resolves
it. A date covered by no version **raises** rather than falling back to today's spec,
because a gap in the history is a data problem to fix, not something to paper over.

```python
specs.add_version("GOLDM", replace(spec, purity=0.999), valid_from=date(2024, 7, 1))
specs.get("GOLDM", on=date(2024, 6, 30)).purity   # 0.995 — listed before
specs.get("GOLDM", on=date(2024, 7, 1)).purity    # 0.999 — listed after
```

Today every symbol has one open-ended version, so nothing changes numerically;
`specs.get(symbol)` with no date still returns the current spec, as every existing
caller expects.

**2. The stored snapshot governs over the registry.** `contract` captures each
contract's spec *as of its listing date*, `read_quotes` joins it back as `spec_*`
columns, and `add_normalized` prefers it. A later edit to the in-code registry
therefore cannot rewrite stored history. Where the two disagree the stored value wins
and `spec_mismatch` is set — the disagreement is a fact worth surfacing (either the
registry was corrected or a revision was never recorded), and silently preferring
either one would hide it.

This is what turned the snapshot from decorative into load-bearing: it was always
written, but nothing read it back.

*A schema-management gap this exposed:* `CREATE TABLE IF NOT EXISTS` does **not**
evolve an existing table — it skips the statement entirely, so a column added to the
definition never reaches a database created before the change, and surfaces later as a
confusing "column does not exist" from unrelated code. `schema.sql` now carries an
explicit idempotent migrations block (`ADD COLUMN IF NOT EXISTS`) after the table
definitions.

### Schema

| Table | Grain | Notes |
|---|---|---|
| `ingest_run` | one load | source, range, row counts, status |
| `bhavcopy_raw` | one upstream row | append-only; unparsed text + `requested_date` |
| `contract` | symbol + expiry | dimension; **spec snapshot as of listing date** |
| `quote` | contract × session | validated fact; all pathology flags retained |
| `dq_flag` | run × issue | the quality ledger |
| `carry_daily` | session | `level` (gold factor), `slope`, fit diagnostics |
| `symbol_premium` | session × symbol | trailing structural premium vs GOLDM |
| `contract_residual` | contract × session | residual, noise floor, tradability |
| `symbol_residual` | session × symbol | aggregated dislocation |
| `pair_signal` | session × pair | z, half-life, ADF, four gates |
| `dq_anomaly` | contract × session | Isolation Forest review queue; **advisory only** |
| `bt_trade` / `bt_daily` | trade / session | backtest output, incl. SPAN margin blocked |
| `bt_position_ledger` | position | full capital record per pair; margin calls, forced exits |
| `risk_param` | date x commodity | published scan range, ELM, additional and delivery margin |
| `risk_spread_credit` | date x leg pair | published cross-margin benefit; absent = no credit |
| `risk_param_file` | date x file | which publication supplied what, and when it was loaded |
| `risk_board` | session x pair | marginal margin, buffer state, add/refuse verdict and its binding reason |

Views: `v_snapshot` (most-liquid tradable contract per symbol), `v_signal_board`
(adds an `action` and names the **`binding_gate`** for anything blocked),
`v_anomaly_review` (flagged rows no rule caught, worst first), `v_echo_audit`,
`v_data_quality`.

There is deliberately **no continuous or near-month series** anywhere in the schema.
Instruments are keyed `symbol|expiry` throughout.

Two storage details worth knowing: an infinite half-life is a legitimate result
("does not revert") but not a storable double, so it lands as `NULL` with
`gate_halflife` carrying the meaning; and `MCX_DB_SCHEMA` is applied by
`repo.init_schema` with identifier quoting rather than hardcoded in the DDL, which is
what lets the test suite isolate itself in a scratch schema instead of touching real
data.

### Getting real MCX data

The current bhavcopy download (`BhavCopyDateWise_DDMMYYYY.csv`) writes dates as
`22 Oct 2026`, headers as `Volume(Lots)` / `Open Interest(Lots)`, and mixes options
(`Instrument Name = OPTFUT`) in with futures. The reader handles all three: only
FUTCOM rows are kept, and repeated downloads of the same day (a browser saves them as
`... (1).csv`, `... (2).csv`) collapse to one session. `tests/test_real_format.py`
runs on eight real rows from the 22 Oct 2026 file.

`www.mcxindia.com` sits behind Akamai bot protection that rejects non-browser TLS
clients regardless of headers — this was verified, and no attempt is made to defeat
it. Three sources are provided behind one interface:

| Source | Use |
|---|---|
| `csv_dir` | **Recommended.** Download bhavcopy CSV/ZIP from MCX in a browser, drop into `data/raw/`, run `--source csv`. No API key. |
| `mcx_live` | The correct JSON client — endpoint, `DD/MM/YYYY` request format, echo validation. Works from a context MCX serves. |
| `synthetic` | Ground-truth generator. Not a substitute for real data; a measuring instrument. |

All three return **raw** bhavcopy shape — MCX's own column names, formats and space
padding — so the synthetic source exercises the identical parsing path, and a test
against it is a real test of production code.

## Data handling

Every pathology becomes a **column**, not a correction. Dropping a bad row hides it;
repairing one invents data. On a 5.7-year synthetic run, 40,555 raw rows became
27,345 validated rows, with:

| Flag | Rows | Handling |
|---|---|---|
| `date_echo_rows` | 13,152 | MCX returned the last session for a holiday/future date — kept under its true date, duplicates collapsed |
| `no_trade_rows` | 6,062 | zero volume ⇒ **derived** settlement; excluded from the curve fit and from fills |
| `symbol_whitespace` | 2,159 | `'GOLDM     '` trimmed before any join |
| `duplicate_key_price_conflict` | 457 | conflicting closes on one (session, contract); most-traded kept |
| `date_flag_ambiguous_ok` | — | below day 13 both orderings parse; resolved against the requested date |

Requests go out `DD/MM/YYYY`; responses come back `MM/DD/YYYY`. The cross-check
against the requested date is what separates a **format bug** from a **date echo** —
without it the two are indistinguishable. Instruments are keyed `symbol|expiry`;
**no continuous or near-month series exists anywhere in the codebase**, because
splicing at a roll injects a jump that a mean-reversion signal reads as opportunity.

## Layout

```
mcxrv/
  specs.py        contract mechanics; quote base vs trading unit; hedge ratios;
                  effective-dated spec versions
  calendar.py     expiry parsing, lifecycle, tender, session extrapolation
  ingest.py       parse + validate; every trap becomes a flag
  normalize.py    -> INR per 10 g fine gold; stored spec beats the registry
  carry.py        two-stage fit: trailing premiums, same-day level and slope
  liquidity.py    tradability, assumed half-spread, measurement noise floor
  signal.py       trailing z-score, ADF, half-life, four gates, execution lag
  eventstudy.py   forward convergence; Newey-West / bootstrap / non-overlap
  costs.py        four crossings, statutory charges, breakeven half-spread
  margin.py       SPAN algorithm, IM floor, ELM minimum, delivery VaR,
                  concentration, volatility schedule, stress scenarios
  ledger.py       per-pair capital ledger, margin calls, forced exits
  risk.py         live risk board: marginal margin, buffer state, add/refuse
  spanfile.py     readers for the published MCX risk parameter files
  backtest.py     walk-forward; integer lots; calendar discipline
  attribution.py  gold beta, roll vs curve, block bootstrap
  anomaly.py      Isolation Forest data-quality review queue (advisory only)
  pipeline.py     steps above as one callable run (shared by scripts and the API)
  snapshot.py     one run -> one JSON file: pairs, reasons, history, risk, backtest
  notify.py       Telegram messages on state changes only
  sources/        base | mcx_live | csv_dir | synthetic | db_source
  db/             config.py, schema.sql, repo.py  (PostgreSQL store)
scripts/          run_analysis.py, make_dashboard.py, db_init.py, db_load.py,
                  nightly.py
api/              service.py (read side, framework-free), main.py (FastAPI routes)
web/              index.html: the dashboard, plain HTML + JS, no build step
tests/            199 tests: null, mutate-the-future, db, ML,
                  spec-versioning, SPAN-margin, ledger, risk-board and
                  published-parameter suites
```

## Margin: SPAN, and the capital denominator

Notional was always tracked. **Capital at risk was not** — returns ran against a
hardcoded ₹20 lakh, which is arbitrary. `mcxrv/margin.py` replaces it.

### What is real and what is assumed

The **SPAN algorithm** is implemented in full: the 16-scenario risk array, the
scanning-risk maximum, intra-commodity (calendar) spread charges, inter-commodity
spread credits, the spot-month charge, and Initial Margin = SPAN + ELM.

The **SPAN parameters are not derivable from bhavcopy.** MCX publishes scan ranges
and spread credits daily in separate risk files, set by its own risk policy — nothing
in OHLC/Volume/OI determines them. So every parameter gets the same treatment as the
bid-ask spread: a documented default, a data-driven fallback
(`derive_scan_range` estimates k·σ from observed EWMA volatility), a sensitivity
sweep, and `SpanParameters.from_mcx_file()` for when real files are available. A
margin number presented as fact when it was assumed is worse than no margin number.

### The initial-margin floor decides everything

MCX gold specifies initial margin as the **higher of ~6 % of contract value and the
SPAN requirement**, with a minimum ELM component and a delivery-period charge computed
as a **5-day 99 % VaR**. Those rules dominate the result.

SPAN does grant a large inter-commodity credit on a fine-gram-matched spread — and
then the floor absorbs all of it:

| | Amount | % of notional |
|---|---|---|
| Scanning risk (both legs) | ₹39,100 | |
| Inter-commodity credit | −₹27,323 | |
| SPAN + ELM computed | ₹22,941 | 3.23 % |
| **6 % floor** | **₹42,654** | **6.00 %** |
| **Blocked** | **₹42,654** | **6.00 %** |

So **the ₹27,323 credit is worth nothing here.** The spread ties up the same 6 % an
outright would while carrying none of the directional upside. Sweep the credit from
0 % to 85 % and SPAN moves from ₹37,325 to ₹15,562 — the margin actually blocked does
not move at all.

| Denominator | Annualised return |
|---|---|
| Hardcoded ₹20 lakh | −0.12 % |
| **Actual margin blocked** (peak ₹78,419) | **−3.09 %** |

*Correcting an earlier claim in this file:* a first pass reported −4.91 % on the basis
that a spread is margin-efficient and therefore looks worse on a small base. With the
specification floor applied, roughly twice as much capital is blocked and the figure
is **−3.09 %**. The direction of the original point still holds against the arbitrary
denominator — a 26× deterioration, not 41× — but the mechanism was wrong: the floor,
not the SPAN credit, sets the capital.

Sharpe is unchanged at −0.80 throughout, because Sharpe is scale-invariant and
therefore says nothing about capital efficiency in either direction.

### Per-pair capital ledger

Every position carries a daily record and a summary row: **notional exposure,
initial margin, SPAN, maintenance, ELM, additional, delivery-period margin, total
capital blocked, daily mark-to-market, margin utilisation, minimum cash buffer, and
any margin call or forced exit.**

Margin is attributed **marginally** — book-with minus book-without — because SPAN
nets and credits across the whole portfolio, so a standalone figure double-counts
risk the book already offsets.

On *maintenance margin*: the US-style level below initial does not exist at MCX.
Indian practice is that the full requirement is maintained at all times, with MTM
settled in **cash** daily and a penalty for shortfall. `maintenance_pct` therefore
defaults to 1.0; it is configurable for venues that work the other way, and the
difference is stated rather than quietly assumed.

### Capital decides the outcome

Because the binding constraint is free cash, the engine simulates margin calls and
liquidates when they go unmet — at a widened spread, since a forced exit crosses a
stressed book. The same signal on the same data:

| | ₹10 lakh account | ₹25,000 account |
|---|---|---|
| Trades | 4 | **8** |
| Margin-call sessions | 0 | 14 |
| Forced liquidations | 0 | **7** |
| Mean net | −40.8 bp | **−53.5 bp** |

Undercapitalisation costs **31 % more per trade**. Note the trade count goes *up*, not
down: a liquidation frees the pair slot while the signal is usually still live, so the
strategy re-enters and is liquidated again. An undercapitalised account churns.

`minimum_viable_capital` reports the smallest account that could have carried the book
without a forced exit — ₹121,215 here, against a peak requirement of ₹78,419 and a
worst drawdown of ₹2,391. "The strategy lost 3 %" means something different on an
account that was never stressed than on one a session from liquidation.

### The live risk board

The ledger above is retrospective. `risk.board()` answers the question a desk asks
*before* trading: given what the book already holds and what margin costs this
morning, is the pair fundable? One row per pair, rendered as a monitor would show it:

```
GOLDM/GOLDTEN spread trade
  Current margin utilization: 78%
  SPAN requirement increased by 14%
  Maintenance buffer: low
  Signal: do not add exposure (maintenance buffer low)
```

Three properties make this a different computation from the ledger:

* **Marginal, not standalone.** The cost of adding a pair is the book's requirement
  with it minus the requirement without it.
* **The requirement moves.** The change is measured over one parameter revision
  cycle with position size held **fixed**, so `+14 %` reports a margin revision and
  never a larger trade.
* **Free cash binds.** A pair that would leave a thin buffer is refused even when the
  margin is nominally affordable and the signal is statistically valid.

`action` exists to say **no**, and it names the binding reason so the refusal can be
argued with: *requirement exceeds capital*, *maintenance buffer low*, *utilisation
above limit*, *SPAN requirement rising; margin regime deteriorating*, *gates not
cleared*, or *lot granularity cannot hedge to tolerance*. A clearing signal never
overrides an unfundable position — that ordering is pinned by a test.

One finding the board makes visible: **while the 6 % floor binds, marginal margin
equals standalone margin.** The floor is a flat percentage of gross notional, and
gross notional is additive, so adding an offsetting pair earns no portfolio credit at
all. The inter-commodity credit SPAN computes is real and entirely unrealised. That
is asserted in `tests/test_risk_board.py` so it cannot silently stop being true.

### Published parameters, not assumed ones

Every margin number is either **published** or **assumed**, and `SpanParameters.source`
says which. Three MCXCCL publications supply the published side:

| File | Supplies |
|---|---|
| **MCX Daily SPAN Risk Parameter File** | price scan ranges, spread charges, spot-month charges, inter-commodity credits |
| **MCXCCL Risk and Collaterals** | ELM, additional/ad-hoc margin, delivery-period margin |
| **MCXCCL Cross-Margin Benefit** | which contract pairs net, and at what rate |

```bash
python scripts/load_risk_params.py --sniff data/risk/span_20260918.txt   # inspect
python scripts/load_risk_params.py --dir data/risk --persist             # load
```

Nothing fetches. MCX refuses programmatic requests and this project does not work
around that, so the files are downloaded by hand into `data/risk/` — the same
arrangement as `csv_dir` for bhavcopy.

**The parser refuses rather than guesses.** A margin rate read out of the wrong column
is worse than no rate at all: it produces a specific, confident, wrong number and
everything downstream inherits it silently. So a file with no business date, a rate
that lands outside 0–1, a header that matches nothing, or a SPAN file that yields no
recognisable scan range all raise `SpanFileError` naming the record — never a
fallback to the default.

Two decisions in that parser worth stating, because getting either wrong is silent and
expensive:

* **The column header decides the units, not the magnitude.** `0.50` in a column
  headed `(%)` is 0.50 %, not 50 %. Inferring from size alone is wrong by a factor of
  a hundred on exactly the small ad-hoc margins that are easiest to overlook.
  Magnitude decides only when the header declares nothing.
* **An unrecognised pair earns zero credit, not the default rate.** Treating "MCX does
  not recognise this spread" as "MCX credits it at 70 %" makes an uncredited two-leg
  position look like a capital-efficient spread. If the cross-margin file does not name
  GOLDM/GOLDPETAL, that spread pays the full scanning risk of both legs — which is a
  *finding about the strategy*, not a gap in the data.

### Dating the parameters is a look-ahead control

A SPAN file describes the risk regime **as of its business date**. Applying one file
across a whole backtest is look-ahead in the margin dimension — the same defect this
project polices in the signal, arriving through a different door — and it flatters the
result precisely when margin was rising and the strategy could least afford it.

Parameters are therefore stored per business date (`risk_param`,
`risk_spread_credit`), looked up for the session being priced, and sessions with no
file keep the **estimated** schedule and stay flagged as estimated. `spanfile.coverage`
reports how much of a backtest actually has published parameters, so "we have real
parameters" and "we have real parameters for the days we tested" cannot be confused.

### Margin is not static

`volatility_schedule` re-estimates scan ranges from trailing EWMA volatility every 21
sessions, so required margin rises when the market is disturbed — which is when a
losing position is least able to fund it. `stress()` prices the deterioration paths:

| Scenario | Margin | vs baseline |
|---|---|---|
| Baseline | ₹42,654 | 1.00 |
| Volatility doubles | ₹42,654 | **1.00** |
| Spread credit halved | ₹42,654 | **1.00** |
| Spread credit withdrawn | ₹46,211 | 1.08 |
| Ad-hoc margin +3 % | ₹49,616 | 1.16 |
| Held into delivery window | ₹65,465 | **1.54** |

An honest and slightly counter-intuitive consequence: **while the floor binds, the
position is insulated from volatility shocks.** You are already posting more than SPAN
asks, so scan-range increases have no effect until something pushes computed margin
above the floor. The risks that do bite are credit withdrawal, ad-hoc margin, and
above all the delivery window — which the engine avoids entirely by force-closing
before tender.

Also modelled: a **concentration surcharge** above 5 % of open interest, which binds
hardest on exactly the thin contracts this strategy wants (a few hundred GOLDPETAL
lots is a large share of that book; the same rupee amount of GOLDM is negligible).

Margin is computed on the **whole open book each session**, not per trade — SPAN nets
deltas and grants credits across everything held, so summing per-trade margins
overstates the requirement. `margin_limit` makes it a real constraint on concurrent
positions rather than the arbitrary `max_concurrent`.

Two structural properties pinned by tests: the ±2× extreme scenarios carry 35 % weight,
so 0.70 < 1.0 and they **can never bind for a linear futures position** (they matter
only once options enter); and a floor on *gross* notional is blind to offsetting
positions, which is why `im_floor_basis` is configurable (`gross` / `net_delta` /
`none`) and the choice is stated rather than assumed.

## Where machine learning is used — and deliberately isn't

scikit-learn is used in exactly two places, both away from the signal. The reason it
is *not* used for the signal is a sample-size fact, not a preference: at |z| ≥ 2 there
are 277 labelled events, but they overlap by 10 sessions, so the **effective
independent sample is ~28**. Nothing beyond a two-parameter model can be fit on that
without memorising it. The carry model's functional form is also derived from
no-arbitrage rather than guessed, and replacing it would forfeit the decomposition
that makes the result defensible — "alpha is 75 % of P&L, roll-down is exactly zero,
gold beta is −0.000" is not a sentence you can say about a boosted tree. And no
forecast fixes a 42.8 bp cost gap.

Both additions are **optional**. Without scikit-learn the carry fit falls back to
weighted least squares and the detector returns nothing; nothing raises.

### 1. Huber regression in the daily carry fit

Plain WLS gives one corrupt settlement unbounded leverage on that day's slope, and
that error propagates into *every* residual for the session. Huber loss bounds it.
Measured with one 300 bp bad print injected into a clean cross-section:

| Solver | Level error | Slope error |
|---|---|---|
| Weighted least squares | **+23.90 bp** | +0.0092 |
| Huber (ε = 1.35) | **+0.09 bp** | +0.0003 |

On *clean* data the two agree to 0.02 bp — robustness costs nothing when there is
nothing to resist. Full-pipeline cost is ~8 % runtime, and slope recovery against
ground truth improved (corr 0.912 → 0.927).

`CarryConfig.robust_scope` controls where the cost is paid: `"day"` (default) is the
same-day level/slope fit only, `"all"` extends to the trailing premium sweep, `"none"`
is an exact escape hatch back to pure WLS. `y` is centred before fitting, because log
prices near 10.8 against τ spanning 0–0.5 make an uncentred intercept numerically
dominant and Huber's joint scale estimate badly conditioned.

*One thing I tried and dropped:* aggregating the premium update with a weighted median
instead of a weighted mean. It measured 6.02 bp recovery error against the mean's
5.88 bp — no better. `c_s` is already an average over ~120 days × ~10 contracts, so a
single bad print moves it by ~1/1200 of its size. Kept as an option, defaulted off.

### 2. Isolation Forest data-quality review queue

`ingest` catches the pathologies someone anticipated well enough to write a rule for.
It cannot catch what nobody anticipated. The forest only needs "unlike the rest of the
data."

Features describe an observation's relation to its own history and to its peers that
session, since a corrupt print is usually wrong relative to one of those rather than
implausible in isolation. The two that carry most of the weight are `ret_vs_peers_bp`
(own move minus the median move of everything else) and `ret_in_ticks` — which is what
separates a real 40 bp GOLDPETAL move (a few ticks) from a decimal-point error.

Validated by injecting price errors that **no existing rule can catch**: OHLC stays
self-consistent, volume is healthy, date and expiry are valid, only the level is
wrong. It found 5 of 6. The miss fell on that contract's first observed session, where
no prior return exists to be anomalous against — no information exists to catch it,
and the test asserts `n-1` rather than pretending otherwise.

On the full store: 27,345 rows scored, 285 flagged, **200 that no rule had caught**.
The worst of those turned out to be a real gap — far-dated contracts with volume of
1–4 lots, 84–117 sessions from expiry, which pass the `no_trade` rule because volume
is technically non-zero.

**These flags are non-blocking and nothing in the model reads them.** That restraint
is load-bearing, and `test_anomaly_flags_never_reach_the_model` enforces it
structurally by asserting no modelling module imports `anomaly`:

- The forest is fit on the whole sample, which is legitimate for a retrospective audit
  but would be flat look-ahead if the flags filtered a backtest — a trade on day *t*
  would be excluded using information from later days.
- Dropping observations a model finds surprising also deletes real market events, and
  those are exactly the days a relative-value strategy lives or dies on. Removing them
  flatters the result.

Run it with `python scripts/db_load.py --model-only --anomaly`; results land in
`dq_anomaly` with the feature vector as `jsonb`, and `v_anomaly_review` surfaces the
unexplained ones worst-first, joined to the quote so a human can see the print.

## Web prototype: API, dashboard, alerts

The research pipeline above answers "is there an edge?". This layer turns the same
pipeline into a monitor someone can open every evening.

```
scripts/nightly.py  ->  data/processed/snapshot.json  ->  api/ (FastAPI)  ->  web/index.html
        |
        +-> Telegram: only when a pair becomes or stops being an ALERT
```

**Nightly job.** `scripts/nightly.py` runs the pipeline once, writes a single JSON
snapshot (atomically, so the API never reads half a file), keeps a dated copy under
`data/processed/history/`, and diffs against the previous snapshot for Telegram.

```bash
python scripts/nightly.py                                   # synthetic sample data
python scripts/nightly.py --source csv --csv-dir data/raw   # downloaded bhavcopy
python scripts/nightly.py --source db --end 2026-09-18      # Postgres store
```

Each pair gets one status for the latest session, and always a reason:

| Status | Meaning |
|---|---|
| ALERT | z beyond ±2 (the entry level), all four gates passed, and the risk board says "may add" or "add reduced size" |
| NO_TRADE | unusual, but something refused: the failed gates are listed, or the risk-board reason |
| QUIET | z inside ±2 |
| WARMUP | not enough history to score yet (the signal needs about 100 sessions per pair) |

The decision log covers recent history, where each unusual session is TRADED
(entered next session), SIGNAL (gates passed but not entered: book full, already
open, or could not size) or NO_TRADE.

**API.** Read-only, served from the snapshot; a request never re-runs the model.

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload        # http://localhost:8000  (docs at /docs)
```

| Endpoint | Returns |
|---|---|
| `GET /api/health` | run id, as-of date, source, sample-data flag |
| `GET /api/overview` | counts, front-month contracts in INR per gram fine, gold level, verdict |
| `GET /api/pairs?status=ALERT` | latest status and reason per pair |
| `GET /api/pairs/GOLDM-GOLDTEN` | one pair: gates, risk board row, 250-session band history |
| `GET /api/decisions?limit=&status=&pair=` | decision log |
| `GET /api/risk` | risk board |
| `GET /api/data-quality` | rows in/out, every flag count, wrong-date examples, data fingerprint |
| `GET /api/backtest` | performance, trades, verdict |

The logic lives in `api/service.py` (no web framework imported) and is what the tests
cover; `api/main.py` is routing only.

**Dashboard.** `web/index.html` is served at `/` by the same app. It is a single file
with no build step, so it can also be hosted on Vercel/Netlify and pointed at a
separate API with `?api=https://your-api.example.com` (set `CORS_ORIGINS` on the API).
A "SAMPLE DATA" badge shows whenever the snapshot came from the synthetic source.

**Telegram.** Create a bot with @BotFather, then set `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` in the environment (or as GitHub secrets). Messages go out only
for a new ALERT, a cleared ALERT, a run with no data, or the same session's data
changing between runs. Without credentials the job prints what it would have sent;
delivery failures never fail the run.

**Automation.** `.github/workflows/nightly.yml` runs the job at 00:15 IST on weekdays,
keeps the previous snapshot in the Actions cache for the diff, and uploads the
snapshot as an artifact. `.github/workflows/ci.yml` runs the test suite with a
Postgres service. `Dockerfile` / `render.yaml` deploy API + dashboard as one service.

**What is not done yet.** The live MCX source still depends on MCX's endpoint
accepting requests from wherever the job runs; on GitHub-hosted runners, download the
bhavcopy separately and use `--source csv`. The dashboard is plain HTML rather than
Next.js so it runs with no toolchain; porting it later changes nothing in the API.

## Known limitations

- **Depth is unobservable.** Bhavcopy has no order book. The half-spread is the
  dominant cost and it is an *assumption*; it is swept, and breakeven half-spread is
  reported instead of a single flattering Sharpe.
- **Settlement ≠ fill.** Flagged where volume is zero, but even a traded day's
  settlement is a closing-window value, not a price anyone was quoted.
- **9 trades is not an inferential sample.** That is why the event study exists on the
  full cross-section; the backtest measures executability, not significance.
- **GOLDTEN has ~1.7 years of history.** Flagged per pair via `years_history`; too few
  independent observations to carry a significance claim.
- **The synthetic generator re-randomises on an echoed day**, so it produces more
  price-conflicting duplicates than real MCX would. This makes the dedup path a
  harder test than reality, which is the intent.
- **Results above are from the synthetic world.** Re-run against downloaded bhavcopy
  before quoting any number as a fact about the real market. The pipeline is
  identical; only the source changes.
- Carry is fit as a single linear slope in τ. Adequate at these maturities; a curved
  term structure would need a second factor.
