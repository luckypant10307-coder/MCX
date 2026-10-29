# Timestamped daily records

One file per MCX session: what the Gold Spread Monitor said (pair status, z-score,
reason, decisions, backtest verdict) and the SHA-256 of the full snapshot. Each
`.json.ots` is an OpenTimestamps proof that the `.json` beside it existed, byte for
byte, by the time of a Bitcoin block. Written automatically by the nightly job.

Verify one record (the proof is pending for a few hours after it is made):

    python scripts/attest.py --verify attestations/<session>_<run>.json

or drop the `.ots` file and the `.json` on https://opentimestamps.org.

A proof shows when the record existed and that it has not changed since. It does not
show that a signal was right or that the prices behind it were correct.
