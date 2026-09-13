# Market Tape data pipeline

This repository builds and publishes the market snapshot consumed by [Market Tape](https://market-tape.cloudandcapital.com/). The frontend is maintained separately in [`cloudandcapital/market-tape-app`](https://github.com/cloudandcapital/market-tape-app).

## Refresh cadence

GitHub Actions checks for updates at `:17` and `:47` during the configured weekday UTC hours (`13–21`). Scheduled events can be delayed or skipped by GitHub; these minutes are not a freshness guarantee. The workflow has a single concurrency group and can also be dispatched manually.

`scripts/refresh_and_publish.sh` fetches the current default-branch tip before building, commits only changed generated data, and retries non-fast-forward races without force-pushing. An unchanged snapshot is a successful no-op.

## Data and validation

Run `python scripts/build_data.py --output-dir /tmp/market-tape-check` to build without replacing tracked snapshots. The build emits `snapshot.json`, `events.json`, `meta.json`, and `mini_rs/` charts. The published equivalents live under `data/`.

`snapshot.json` contains seven displayed groups and 52 core rows. Each rendered price carries `price_date`, `price_source`, and `price_status`. A row is current only when its price date matches the SPY benchmark session; missing or stale rows fail snapshot validation. `meta.json` records the generation time, counts, market status, and screened-universe leaderboard. The leaderboard excludes unavailable or stale candidates rather than silently publishing them.

Install dependencies with `python -m pip install -r requirements.txt`, then run `python -m unittest discover -s tests -v`. The workflow and tests cover schedule semantics, freshness, output ordering, and safe publication behavior.

The frontend fetches the published JSON from the repository and revalidates on its own cache cadence. A successful pipeline run does not by itself mean a market session is newer; check `meta.json` and row-level price dates before describing the dashboard as fresh.
