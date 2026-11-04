# Experiments: can the system decide when a trade is worth its cost?

*Run on 4 November 2026, on real MCX data from January 2022 to October 2026
(1,237 sessions).*

## The problem

The backtest makes **+17.5 bp per trade before costs and −5.3 bp after**. The gaps the
system finds are real and they do close; the trades lose because trading costs more
than they earn. The entry rule only asks whether a gap is unusual (|z| ≥ 2). It never
asks whether the gap is big enough to pay for the trade.

## The method, fixed before looking at results

- Every rule is chosen using trades **entered in 2022–2024** only.
- It is then read once on trades **entered in 2025–2026**, which played no part in
  choosing it.
- A learned model must **beat a simple rule** on the unseen years to be used.

## 1. A cost-aware entry gate

Before entering, estimate how many bp the gap will close, compute this trade's actual
round-trip cost (these contracts, these lots, today's spread estimates, brokerage and
exchange charges, four crossings), and enter only if gain − cost exceeds a margin.

Two ways to estimate the gain:

- **Rule:** assume the gap closes all the way to the exit band,
  |gap| × (1 − 0.5 / |z|).
- **Learned:** a ridge regression on gap size, |z|, the pair's half-life and whether
  the contracts share an expiry window, refitted every month on outcomes that were
  already known by then ([`mcxrv/stage3_gates_decision/edge.py`](../mcxrv/stage3_gates_decision/edge.py)).

| Version | All trades | 2022–24 net / trade | 2025–26 net / trade |
|---|---|---|---|
| Current system | 146 | −2.1 bp | −7.6 bp |
| Rule gain, margin 10 bp (best on 2022–24) | 144 | −1.4 bp | −7.4 bp |
| Learned gain, margin 0 (best on 2022–24) | 135 | −5.7 bp | −6.4 bp |
| Learned gain, margin 10 bp | 124 | −5.7 bp | −15.5 bp |

**Result: no improvement.** The rule barely filters, because it assumes nearly every
gap closes fully. The learned model ranks gaps poorly (rank correlation with the real
outcome 0.28 on 2025–26, against 0.39 for the rule) and was too optimistic: it
expected about 34 bp of closing where 19 bp happened. Gaps have closed less since 2025,
and a model trained on earlier years cannot know that in advance.

**Decision:** the learned model does not beat the rule, so it is not used. The gate
stays in the backtest, switched off (`BacktestConfig.min_edge_bp = None`).

## 2. Trading only the widest gaps

| Entry rule | 2022–24 | 2025–26 |
|---|---|---|
| \|z\| ≥ 2.0 (current) | 62 trades, −2.1 bp | 84 trades, −7.6 bp |
| \|z\| ≥ 2.5 | 46 trades, −10.8 bp | 78 trades, +0.5 bp |
| \|z\| ≥ 3.0 | 37 trades, −7.2 bp | 64 trades, **+4.2 bp** |
| \|z\| ≥ 3.5 | 35 trades, −4.6 bp | 58 trades, +2.5 bp |

**Result: not proven.** On 2022–24, the years the rule must be chosen from, every
threshold loses and the current one loses least. Picking |z| ≥ 3 because it looks good
on 2025–26 would be choosing the rule after seeing the answer. 64 trades at +4.2 bp is
also small enough to be luck.

## 3. A cost hurdle from the gap's own half-life

*Run on 4 November 2026. Settings fixed before running and not tuned: haircut 0.7,
gain at least 1.5x cost, at least 5 bp net.*

[`edge_gate.edge_check`](../mcxrv/stage3_gates_decision/edge_gate.py) projects how far
the gap closes before the trade must end, if it decays at its measured half-life:

    expected gross = (|z| - 0.5) x scale x (1 - 0.5 ^ (H / half-life)) x 0.7

and compares it with this trade's entry cost plus an estimated exit cost (wider on a
leg that is already untradable). The same function decides the backtest's entries
and, when switched on, the live alerts, where a pair that fails it is shown as WATCH.

Mean bp per trade, with a 95% moving-block bootstrap interval (blocks of 5 trades in
entry order; 2,000 resamples):

| Brokerage | Gate | 2022–24 net | 2025–26 net |
|---|---|---|---|
| floor (2 bp, ₹20 minimum) | off | −2.1 [−18.5, +11.9], 62 trades | −7.6 [−35.6, +23.6], 84 |
| floor | on | −3.0 [−21.8, +12.0], 54 | −8.2 [−37.0, +22.7], 83 |
| cap (0.03% or ₹20, lower) | off | **+5.4** [−11.0, +19.4], 62 | **+1.1** [−26.8, +32.4], 84 |
| cap | on | +6.3 [−12.2, +22.1], 57 | +1.1 [−26.1, +32.2], 84 |

**The hurdle does not help.** It refuses 8 trades in 2022–24 and 1 in 2025–26, and
the net does not improve. Its expected value does not rank trades either: split into
thirds by expected net (cut points from 2022–24), the top third did worst in 2022–24
(−13.9 bp) and best in 2025–26 (−0.8 bp), the opposite order. **It stays switched
off** (`PipelineConfig.cost_gate = False`).

**Brokerage matters more than any entry rule.** "Floor" charges about 8 bp of
brokerage per round trip; "cap", which is how discount brokers such as Zerodha charge
MCX futures (confirmed by Kite Connect's contract-note example), charges under 2 bp.
Under "cap" the average trade is slightly positive, but every interval includes zero
by a wide margin: it is indistinguishable from breaking even, not an edge.

Where the money goes (floor brokerage, gate off, all years): trades that converged
made +29.9 bp (47 trades); stops lost −22.2 bp (75) and expiry exits −21.3 bp (24).
GOLDGUINEA/GOLDPETAL lost −20.9 bp a trade (35); only GOLDM/GOLDGUINEA was positive
(+6.2 bp, 44). The bid/ask estimate fell from 16 bp a trade in 2022–24 to 5 bp in
2025–26 as volumes grew.

Half-spread sensitivity, reported and not chosen from (floor brokerage, net bp):

| Half-spread × | Gate off 2022–24 | Gate off 2025–26 | Gate on 2022–24 | Gate on 2025–26 |
|---|---|---|---|---|
| 0.5 | +0.6 | −6.4 | +0.2 | −6.6 |
| 1.0 | −2.1 | −7.6 | −3.0 | −8.2 |
| 1.5 | −4.8 | −8.8 | −8.0 | −9.4 |
| 2.0 | −7.6 | −10.1 | −9.5 | −10.3 |

## 4. What was kept

- **Cost on every alert.** Each pair now shows the round-trip cost of the trade the
  risk board would put on, next to the gap: two measured numbers, no model. On
  27 October 2026 a round trip cost about 16–17 bp, of which brokerage and exchange
  charges were about 13 bp and the bid/ask spread 3–4 bp.
- **A pre-registered forward test.** From 5 November 2026 the system runs |z| ≥ 3 next
  to the current rule every night and counts only trades entered from that date, on
  sessions that did not exist when the rule was chosen. Alerts are unchanged. Results
  are on the Research page. The rule and start date are fixed in
  `mcxrv/pipeline.py` (`FORWARD_TEST`) and must not be changed once results arrive.

## Reproducing

```bash
python scripts/fetch_mtf.py
python scripts/experiments/edge_experiment.py     # section 1
python scripts/experiments/zentry_experiment.py   # section 2
python scripts/experiments/cost_hurdle_eval.py    # section 3
```

Numbers will move slightly as new sessions are added.
