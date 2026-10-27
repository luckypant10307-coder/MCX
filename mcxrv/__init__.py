"""
MCX Gold Cross-Contract Relative Value, organised by pipeline stage.

  stage1_intake            INTAKE & CHECKS      specs, calendar, validation, anomaly, store
  stage2_normalize_signal  MODEL & SIGNAL       normalize, carry fit, z-score, liquidity
  stage3_gates_decision    GATES & DECISION     costs, margin, risk board, risk files
  stage4_simulate_deliver  SIMULATE & EVALUATE  replay, ledger, attribution, event study

Import a module from its stage, e.g.
`from mcxrv.stage3_gates_decision import margin_capital as margin`.

Top level: `pipeline` (the whole run as one call), `snapshot` (what the API serves),
and the delivery side: `notify`, `user_notify`, `supa`, `circulars`, `risk_inbox`,
`published_margins`.

`stage1_intake.canonical` is the Postgres layer and needs psycopg; nothing imports it
unless a database is in use.
"""
