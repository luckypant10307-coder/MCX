-- MCX gold relative-value store.
--
-- Layering principle: `bhavcopy_raw` is APPEND-ONLY and immutable -- exactly what the
-- upstream returned, including its mistakes. Everything else is DERIVED and can be
-- rebuilt from it. Validation rules and model parameters will change; the raw record
-- must not, or you lose the ability to re-run history under a corrected rule.
--
-- The single most important column here is `bhavcopy_raw.requested_date`. MCX answers
-- an unrecognised, holiday, malformed or future date with the most recent available
-- trading day. If only the ECHOED date is stored, that substitution becomes
-- undetectable forever after: one session silently duplicates across many dates,
-- manufacturing zero-return days and inflating every Sharpe computed downstream.
-- Keeping both dates is what preserves the ability to detect it.
--
-- All derived tables upsert on their natural key, so a reload is idempotent and a
-- daily append is safe to re-run.

-- The schema itself is created and put on the search_path by `repo.init_schema`,
-- from MCX_DB_SCHEMA, with proper identifier quoting. It is deliberately NOT named
-- here: hardcoding it would make the setting a lie and would stop the test suite
-- isolating itself in a scratch schema.

-- ---------------------------------------------------------------- provenance

CREATE TABLE IF NOT EXISTS ingest_run (
    run_id          bigserial PRIMARY KEY,
    source          text        NOT NULL,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    range_start     date,
    range_end       date,
    rows_fetched    integer,
    rows_validated  integer,
    status          text        NOT NULL DEFAULT 'running',
    note            text,
    CONSTRAINT ingest_run_status_ck
        CHECK (status IN ('running', 'ok', 'failed'))
);

COMMENT ON TABLE ingest_run IS
    'One row per load. Makes every stored row traceable to when and how it arrived.';

-- ------------------------------------------------------------------- raw tape

CREATE TABLE IF NOT EXISTS bhavcopy_raw (
    raw_id          bigserial PRIMARY KEY,
    run_id          bigint      REFERENCES ingest_run(run_id) ON DELETE SET NULL,
    source          text        NOT NULL,
    fetched_at      timestamptz NOT NULL DEFAULT now(),
    -- the date WE ASKED FOR, in our own canonical form
    requested_date  date,
    -- the fields exactly as the upstream rendered them, unparsed
    raw_date        text,
    raw_symbol      text,
    raw_expiry      text,
    raw_open        text,
    raw_high        text,
    raw_low         text,
    raw_close       text,
    raw_volume      text,
    raw_open_interest text,
    payload         jsonb
);

CREATE INDEX IF NOT EXISTS bhavcopy_raw_req_idx  ON bhavcopy_raw (requested_date);
CREATE INDEX IF NOT EXISTS bhavcopy_raw_run_idx  ON bhavcopy_raw (run_id);
CREATE INDEX IF NOT EXISTS bhavcopy_raw_sym_idx  ON bhavcopy_raw (raw_symbol);

COMMENT ON COLUMN bhavcopy_raw.requested_date IS
    'The date requested. Compared against the parsed raw_date to detect MCX''s '
    'date echo. Without this column the echo is undetectable after the fact.';
COMMENT ON COLUMN bhavcopy_raw.raw_date IS
    'Unparsed. MCX responds MM/DD/YYYY while requests go out DD/MM/YYYY.';

-- ----------------------------------------------------------------- dimension

CREATE TABLE IF NOT EXISTS contract (
    contract_id     text PRIMARY KEY,          -- 'GOLDM|2026-09-04'
    symbol          text        NOT NULL,
    expiry_date     date        NOT NULL,
    trading_unit_g  numeric(12,4) NOT NULL,
    quote_base_g    numeric(12,4) NOT NULL,
    purity          numeric(6,4)  NOT NULL,
    fine_g_per_lot  numeric(12,4) NOT NULL,
    tick_size_inr   numeric(12,4),
    deliverable     text,
    first_seen      date,
    last_seen       date,
    expiry_window_ok boolean,
    UNIQUE (symbol, expiry_date)
);

CREATE INDEX IF NOT EXISTS contract_symbol_idx ON contract (symbol, expiry_date);

COMMENT ON TABLE contract IS
    'Instruments keyed by symbol + expiry. There is deliberately NO continuous or '
    'near-month series anywhere in this schema: splicing contracts at a roll injects '
    'an artificial price jump that a mean-reversion signal reads as an opportunity.';

-- ------------------------------------------------------------------ the fact

CREATE TABLE IF NOT EXISTS quote (
    contract_id     text NOT NULL REFERENCES contract(contract_id) ON DELETE CASCADE,
    trade_date      date NOT NULL,
    symbol          text NOT NULL,
    expiry_date     date NOT NULL,

    open            numeric(18,4),
    high            numeric(18,4),
    low             numeric(18,4),
    close           numeric(18,4) NOT NULL,
    volume          bigint NOT NULL DEFAULT 0,
    open_interest   bigint NOT NULL DEFAULT 0,

    -- normalised to the common basis: INR per 10 g of FINE gold
    norm_close      numeric(18,6),

    -- lifecycle
    days_to_expiry      integer,
    sessions_to_expiry  integer,
    sessions_estimated  boolean DEFAULT false,
    t_years             numeric(10,6),
    in_tender           boolean DEFAULT false,

    -- validation flags: pathologies are recorded, never silently corrected
    date_flag           text,
    date_echo           boolean DEFAULT false,
    expiry_window_bad   boolean DEFAULT false,
    ohlc_bad            boolean DEFAULT false,
    no_trade            boolean DEFAULT false,
    settle_untradeable  boolean DEFAULT false,
    spec_mismatch       boolean DEFAULT false,

    run_id          bigint REFERENCES ingest_run(run_id) ON DELETE SET NULL,
    updated_at      timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (contract_id, trade_date),
    CONSTRAINT quote_close_positive_ck CHECK (close > 0)
);

CREATE INDEX IF NOT EXISTS quote_date_idx     ON quote (trade_date);
CREATE INDEX IF NOT EXISTS quote_sym_date_idx ON quote (symbol, trade_date);
CREATE INDEX IF NOT EXISTS quote_tradeable_idx ON quote (trade_date)
    WHERE NOT settle_untradeable;

COMMENT ON COLUMN quote.close IS
    'Official exchange settlement. NOT an executable fill price -- on a no-trade day '
    'it is derived outright (see no_trade / settle_untradeable).';
COMMENT ON COLUMN quote.norm_close IS
    'close * 10 / (quote_base_g * purity). Uses the QUOTATION BASE, never the trading '
    'unit: GOLDM has a 100 g unit but is quoted per 10 g, so it already shares '
    'GOLDTEN''s base and differs only in purity.';

-- ------------------------------------------------------- data quality ledger

CREATE TABLE IF NOT EXISTS dq_flag (
    run_id      bigint NOT NULL REFERENCES ingest_run(run_id) ON DELETE CASCADE,
    issue       text   NOT NULL,
    rows        integer NOT NULL,
    detail      text,
    PRIMARY KEY (run_id, issue)
);

COMMENT ON TABLE dq_flag IS
    'Per-run tally of every pathology found. Kept so a report can state how much data '
    'it stood on rather than implying the input was clean.';

-- ------------------------------------------------------------ model outputs

CREATE TABLE IF NOT EXISTS carry_daily (
    trade_date      date PRIMARY KEY,
    level           double precision,   -- a_t: the gold factor
    slope           double precision,   -- b_t: annualised carry
    fit_rmse        double precision,
    n_obs           integer,
    n_window_days   integer,
    has_premium_est boolean,
    d_level         double precision,
    d_slope         double precision,
    model_run_at    timestamptz NOT NULL DEFAULT now()
);

COMMENT ON COLUMN carry_daily.level IS
    'Fitted log level of fine gold. Its first difference is the gold factor used for '
    'beta attribution -- derived from the same cross-section as everything else.';

CREATE TABLE IF NOT EXISTS symbol_premium (
    trade_date      date NOT NULL,
    symbol          text NOT NULL,
    premium_bp      double precision,
    PRIMARY KEY (trade_date, symbol)
);

COMMENT ON TABLE symbol_premium IS
    'Structural premium vs the GOLDM 995 bar, estimated from a TRAILING window only. '
    'Represents fabrication cost embedded in a minted deliverable -- persistent and '
    'economically justified, not a mispricing.';

CREATE TABLE IF NOT EXISTS contract_residual (
    contract_id     text NOT NULL REFERENCES contract(contract_id) ON DELETE CASCADE,
    trade_date      date NOT NULL,
    residual_bp     double precision,
    fitted_log      double precision,
    fit_weight      double precision,
    in_fit          boolean,
    premium_bp      double precision,
    noise_bp        double precision,
    half_spread_bp  double precision,
    max_lots        double precision,
    tradable        boolean,
    PRIMARY KEY (contract_id, trade_date)
);

CREATE INDEX IF NOT EXISTS contract_residual_date_idx ON contract_residual (trade_date);

CREATE TABLE IF NOT EXISTS symbol_residual (
    trade_date      date NOT NULL,
    symbol          text NOT NULL,
    resid_bp        double precision,
    n               integer,
    wsum            double precision,
    PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS pair_signal (
    trade_date      date NOT NULL,
    pair            text NOT NULL,
    sym_a           text NOT NULL,
    sym_b           text NOT NULL,
    maturity_matched boolean,
    spread_bp       double precision,
    z               double precision,
    dislocation_bp  double precision,
    centre          double precision,
    scale           double precision,
    pctile          double precision,
    halflife        double precision,
    adf_t           double precision,
    trail_n         double precision,
    gate_history    boolean,
    gate_stationary boolean,
    gate_halflife   boolean,
    gate_magnitude  boolean,
    signal_ok       boolean,
    direction       smallint,
    PRIMARY KEY (trade_date, pair)
);

CREATE INDEX IF NOT EXISTS pair_signal_live_idx ON pair_signal (trade_date)
    WHERE signal_ok;

COMMENT ON TABLE pair_signal IS
    'Every statistic is trailing: means, deviations, half-lives and ADF tests use data '
    'strictly before trade_date. A signal here is not yet executable -- it can only be '
    'acted on the FOLLOWING session, because a settlement is published after the close.';

-- ------------------------------------------------- anomaly review queue

CREATE TABLE IF NOT EXISTS dq_anomaly (
    contract_id     text NOT NULL REFERENCES contract(contract_id) ON DELETE CASCADE,
    trade_date      date NOT NULL,
    symbol          text NOT NULL,
    score           double precision,   -- sklearn decision_function; LOWER = worse
    is_anomaly      boolean,
    top_feature     text,               -- which feature was most extreme
    top_z           double precision,   -- its robust (MAD) deviation
    rule_flagged    boolean,            -- did ingest already catch this by rule?
    features        jsonb,
    detected_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (contract_id, trade_date)
);

CREATE INDEX IF NOT EXISTS dq_anomaly_flagged_idx ON dq_anomaly (trade_date)
    WHERE is_anomaly;

COMMENT ON TABLE dq_anomaly IS
    'Unsupervised review queue from an Isolation Forest. NON-BLOCKING by design: '
    'nothing in carry, signal or backtest reads it. The forest is fit on the whole '
    'sample, which is legitimate for a retrospective data audit but would be flat '
    'look-ahead if these flags ever filtered a backtest -- a trade on day t would '
    'then be excluded using information from later days. Dropping surprising rows '
    'also deletes real market events, which are exactly the days a relative-value '
    'strategy lives or dies on.';
COMMENT ON COLUMN dq_anomaly.rule_flagged IS
    'True where ingest had already caught the row. The value of this table is in the '
    'rows where this is FALSE: pathologies nobody wrote a rule for.';

-- ----------------------------------------------------------------- backtest

CREATE TABLE IF NOT EXISTS bt_trade (
    trade_id        bigserial PRIMARY KEY,
    model_run_id    bigint,
    pair            text NOT NULL,
    sym_a           text, sym_b text,
    cid_a           text, cid_b text,
    dir_pair        smallint,
    lots_a          integer, lots_b integer,
    entry_date      date NOT NULL,
    exit_date       date NOT NULL,
    sessions_held   integer,
    entry_z         double precision,
    exit_z          double precision,
    entry_spread_bp double precision,
    exit_spread_bp  double precision,
    notional_a      double precision,
    notional_b      double precision,
    fine_residual_g double precision,
    residual_frac   double precision,
    gross_pnl       double precision,
    total_cost      double precision,
    net_pnl         double precision,
    gross_bp        double precision,
    net_bp          double precision,
    exit_reason     text,
    illiquid_exit   boolean
);

CREATE INDEX IF NOT EXISTS bt_trade_pair_idx ON bt_trade (pair, entry_date);

COMMENT ON COLUMN bt_trade.residual_frac IS
    'Unhedged fine-gram fraction from whole-lot sizing. Never zero: GOLDM:GOLDGUINEA '
    'is 24875/1998 in fine grams, so the smallest exactly neutral position is '
    '1998 x 24875 lots. Recorded so gold leak can be attributed rather than assumed away.';

CREATE TABLE IF NOT EXISTS bt_daily (
    model_run_id    bigint NOT NULL,
    trade_date      date   NOT NULL,
    realised_cum    double precision,
    unrealised      double precision,
    equity          double precision,
    day_realised    double precision,
    day_cost        double precision,
    n_open          integer,
    ret             double precision,
    PRIMARY KEY (model_run_id, trade_date)
);

-- ---------------------------------------------------------------- migrations
--
-- CREATE TABLE IF NOT EXISTS does NOT evolve a table that already exists: it skips
-- the statement entirely, so a column added to the definition above never reaches a
-- database created before the change, and the failure surfaces later as a confusing
-- "column does not exist" from unrelated code. Every column added after the first
-- release therefore needs an explicit, idempotent ALTER here.
--
-- ADD COLUMN IF NOT EXISTS is safe to re-run, so this block is part of the ordinary
-- schema application rather than a separate migration step.

ALTER TABLE contract   ADD COLUMN IF NOT EXISTS tick_size_inr      numeric(12,4);
ALTER TABLE quote      ADD COLUMN IF NOT EXISTS sessions_estimated boolean DEFAULT false;
ALTER TABLE quote      ADD COLUMN IF NOT EXISTS spec_mismatch      boolean DEFAULT false;
ALTER TABLE bt_trade   ADD COLUMN IF NOT EXISTS entry_margin       double precision;
ALTER TABLE bt_trade   ADD COLUMN IF NOT EXISTS return_on_margin   double precision;
ALTER TABLE bt_daily   ADD COLUMN IF NOT EXISTS margin             double precision;
ALTER TABLE bt_daily   ADD COLUMN IF NOT EXISTS span               double precision;
ALTER TABLE bt_daily   ADD COLUMN IF NOT EXISTS inter_credit       double precision;

-- Per-pair capital ledger. One row per position: what it cost to open, what it cost
-- to HOLD, how bad the mark-to-market got, and whether the account was ever forced
-- out. Indian clearing settles MTM in cash daily, so free cash -- not notional and
-- not even margin -- is what decides whether a position can be carried.
CREATE TABLE IF NOT EXISTS bt_position_ledger (
    model_run_id            bigint NOT NULL,
    trade_key               text   NOT NULL,
    pair                    text,
    entry_date              date,
    exit_date               date,
    sessions_held           integer,
    notional_exposure       double precision,
    initial_margin          double precision,
    initial_margin_peak     double precision,
    span_margin             double precision,
    span_margin_peak        double precision,
    maintenance_margin      double precision,
    elm                     double precision,
    additional_margin       double precision,
    delivery_margin         double precision,
    concentration_margin    double precision,
    inter_commodity_credit  double precision,
    floor_binds             boolean,
    total_capital_blocked   double precision,
    margin_pct_notional     double precision,
    peak_margin_utilisation double precision,
    mtm_final               double precision,
    mtm_worst               double precision,
    mtm_best                double precision,
    mtm_daily_vol           double precision,
    min_cash_buffer_required double precision,
    margin_call_sessions    integer,
    was_forced_out          boolean,
    net_pnl                 double precision,
    return_on_capital_blocked double precision,
    exit_reason             text,
    PRIMARY KEY (model_run_id, trade_key)
);

CREATE INDEX IF NOT EXISTS bt_position_ledger_forced_idx
    ON bt_position_ledger (model_run_id) WHERE was_forced_out;

COMMENT ON COLUMN bt_position_ledger.min_cash_buffer_required IS
    'Worst cumulative mark-to-market drawdown during the position. Free cash below '
    'this would have produced a margin call and, unmet, a forced liquidation.';
COMMENT ON COLUMN bt_position_ledger.was_forced_out IS
    'Liquidated because the account could not fund the book. A forced exit crosses '
    'a stressed market and costs more than a chosen one, so a strategy that only '
    'works when never forced out is not a strategy.';

COMMENT ON COLUMN contract.tick_size_inr IS
    'Part of the spec snapshot: it sets tick_bp_of_norm and hence the measurement '
    'noise floor the signal gate compares against.';
COMMENT ON COLUMN bt_daily.margin IS
    'SPAN + ELM blocked by the whole open book at the close, recomputed daily. '
    'Margin is a portfolio property -- SPAN nets offsetting deltas and grants '
    'inter-commodity credits across everything held -- so summing per-trade margins '
    'would overstate it, often several-fold.';
COMMENT ON COLUMN bt_daily.inter_credit IS
    'Credit waived for offsetting related commodities. A spread is margin-EFFICIENT, '
    'so the same rupee loss sits on a far smaller capital base than the notional '
    'suggests: correct margin accounting makes a losing spread look WORSE.';
COMMENT ON COLUMN quote.spec_mismatch IS
    'The stored per-contract specification disagreed with the in-code registry. '
    'The STORED value governs, so history is never rewritten by a code change; this '
    'flag exists so the disagreement is visible rather than silently resolved.';

-- -------------------------------------------------------------------- views

CREATE OR REPLACE VIEW v_latest_session AS
SELECT max(trade_date) AS trade_date FROM quote;

CREATE OR REPLACE VIEW v_snapshot AS
-- Most-liquid tradable contract per symbol on the latest session.
SELECT DISTINCT ON (q.symbol)
       q.trade_date, q.symbol, q.contract_id, q.expiry_date,
       q.close, q.norm_close, q.volume, q.open_interest,
       q.sessions_to_expiry, q.in_tender,
       r.residual_bp, r.premium_bp, r.noise_bp, r.half_spread_bp,
       r.max_lots, r.tradable, q.t_years
FROM quote q
JOIN v_latest_session ls ON q.trade_date = ls.trade_date
LEFT JOIN contract_residual r
       ON r.contract_id = q.contract_id AND r.trade_date = q.trade_date
WHERE NOT q.settle_untradeable
ORDER BY q.symbol, COALESCE(r.tradable, false) DESC, q.volume DESC;

COMMENT ON VIEW v_snapshot IS
    'Chosen by volume, not by nearness to expiry: the leg you could actually execute '
    'is the one with a tape, and the front month is not always it.';

CREATE OR REPLACE VIEW v_signal_board AS
SELECT s.*,
       CASE WHEN s.signal_ok THEN
            CASE WHEN s.direction > 0 THEN 'buy ' || s.pair
                 ELSE 'sell ' || s.pair END
            ELSE 'hold' END AS action,
       CASE WHEN NOT s.gate_history    THEN 'history'
            WHEN NOT s.gate_stationary THEN 'not stationary'
            WHEN NOT s.gate_halflife   THEN 'half-life too long'
            WHEN NOT s.gate_magnitude  THEN 'below noise floor'
            ELSE NULL END AS binding_gate
FROM pair_signal s;

COMMENT ON VIEW v_signal_board IS
    'Names the gate that blocks each pair. A dashboard that says WHY it is quiet is '
    'more trustworthy than one that only says it has nothing.';

CREATE OR REPLACE VIEW v_data_quality AS
SELECT r.run_id, r.source, r.started_at, r.range_start, r.range_end,
       r.rows_fetched, r.rows_validated, r.status,
       f.issue, f.rows AS issue_rows, f.detail
FROM ingest_run r
LEFT JOIN dq_flag f USING (run_id);

CREATE OR REPLACE VIEW v_anomaly_review AS
-- The review queue that matters: flagged rows no rule anticipated, worst first,
-- joined to the quote so a human can see the print itself.
SELECT a.trade_date, a.symbol, a.contract_id, a.score, a.top_feature, a.top_z,
       q.close, q.norm_close, q.volume, q.open_interest, q.sessions_to_expiry
FROM dq_anomaly a
JOIN quote q ON q.contract_id = a.contract_id AND q.trade_date = a.trade_date
WHERE a.is_anomaly AND NOT a.rule_flagged
ORDER BY a.score;

CREATE OR REPLACE VIEW v_echo_audit AS
-- Rows where MCX answered with a session other than the one requested.
SELECT requested_date, raw_date, count(*) AS rows
FROM bhavcopy_raw
WHERE requested_date IS NOT NULL
  AND raw_date IS NOT NULL
  AND to_date(raw_date, 'MM/DD/YYYY') <> requested_date
GROUP BY requested_date, raw_date
ORDER BY requested_date;

-- Live risk board. One row per pair per session evaluated: what adding that pair
-- would cost MARGINALLY against the book already open, where that leaves the account,
-- how the requirement has moved over a parameter revision cycle, and the resulting
-- instruction. Stored because the instruction is only auditable after the fact if the
-- numbers behind it were kept: "do not add exposure" is a decision, and a decision
-- without its inputs cannot be reviewed.
CREATE TABLE IF NOT EXISTS risk_board (
    model_run_id           bigint NOT NULL,
    asof                   date   NOT NULL,
    pair                   text   NOT NULL,
    account_capital        double precision,
    lots_a                 integer,
    lots_b                 integer,
    direction              integer,
    notional_exposure      double precision,
    span_margin            double precision,
    initial_margin         double precision,
    elm                    double precision,
    delivery_margin        double precision,
    concentration_margin   double precision,
    inter_commodity_credit double precision,
    floor_binds            boolean,
    marginal_margin        double precision,
    book_required          double precision,
    required_if_added      double precision,
    free_cash_if_added     double precision,
    margin_utilisation     double precision,
    utilisation_if_added   double precision,
    buffer_mult            double precision,
    buffer_state           text,
    span_change_pct        double precision,
    im_change_pct          double precision,
    scan_range_pct         double precision,
    scan_range_prior_pct   double precision,
    lookback_from          date,
    z                      double precision,
    spread_bp              double precision,
    signal_ok              boolean,
    sessions_to_expiry     integer,
    action                 text,
    reason                 text,
    PRIMARY KEY (model_run_id, asof, pair)
);
CREATE INDEX IF NOT EXISTS risk_board_blocked_idx
    ON risk_board (asof) WHERE action = 'do not add exposure';

COMMENT ON COLUMN risk_board.marginal_margin IS
    'Book requirement WITH this pair minus the requirement WITHOUT it. Equals the '
    'standalone margin whenever the initial-margin floor binds, because the floor is '
    'a flat percentage of gross notional and gross notional is additive.';
COMMENT ON COLUMN risk_board.span_change_pct IS
    'Change in the SPAN requirement over lookback_from .. asof with position size '
    'held FIXED, so it reports a margin revision rather than a larger trade.';
COMMENT ON COLUMN risk_board.reason IS
    'The BINDING constraint, not a generic one. A refusal that does not name what '
    'blocked it cannot be argued with.';

-- ---------------------------------------------------------------------------
-- Published MCX risk parameters
-- ---------------------------------------------------------------------------
-- Every margin number the model uses is either PUBLISHED or ASSUMED, and a report
-- that cannot tell them apart is not worth much. These tables hold the published
-- side, keyed by the business date the file was issued for.
--
-- The date is load-bearing, not bookkeeping. A SPAN file describes the risk regime
-- AS OF its business date; applying today's file across a historical backtest is
-- look-ahead in the margin dimension -- the same defect this project polices in the
-- signal, arriving through a different door, and it flatters the result exactly when
-- margin was rising. Parameters are therefore looked up per session and the model
-- falls back to its ESTIMATED schedule, flagged, wherever a file is missing.

CREATE TABLE IF NOT EXISTS risk_param_file (
    business_date date NOT NULL,
    kind          text NOT NULL,      -- 'span' | 'risk_collateral' | 'cross_margin'
    source_path   text NOT NULL,
    exchange      text,
    commodities   integer,
    credits       integer,
    unrecognised  integer,
    notes         text,
    loaded_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (business_date, kind, source_path)
);

CREATE TABLE IF NOT EXISTS risk_param (
    business_date       date NOT NULL,
    commodity           text NOT NULL,
    scan_pct            double precision,
    scan_basis          text,
    intra_spread_charge double precision,
    spot_month_charge   double precision,
    elm_pct             double precision,
    additional_pct      double precision,
    delivery_pct        double precision,
    source_path         text,
    PRIMARY KEY (business_date, commodity)
);

CREATE TABLE IF NOT EXISTS risk_spread_credit (
    business_date date NOT NULL,
    leg_a         text NOT NULL,
    leg_b         text NOT NULL,
    credit_rate   double precision,
    ratio_a       integer,
    ratio_b       integer,
    priority      integer,
    source_path   text,
    PRIMARY KEY (business_date, leg_a, leg_b)
);

COMMENT ON TABLE risk_spread_credit IS
    'MCXCCL cross-margin benefit: which contract pairs net, and by how much. For a '
    'cross-contract gold strategy this is the single most consequential published '
    'number -- every claim about capital efficiency rests on it. A pair ABSENT here '
    'earns no credit; treating absence as the default rate would turn "not '
    'recognised" into "generously credited".';
COMMENT ON COLUMN risk_param.scan_basis IS
    'pct = fraction of contract value; inr_per_unit = currency per contract, which '
    'needs the contract value before it can be compared across contract sizes.';
