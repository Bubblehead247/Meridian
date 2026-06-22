# Phase 1 Summary — Data Engineering

**Status:** Complete
**Date:** 2026-06-22

Phase 1 builds `meridian/data/`: the only path the platform uses to obtain
price history. It covers universe management, yfinance ingestion, on-disk
caching, and the out-of-sample date splitting that enforces evaluation
discipline. Everything is offline-testable; the live yfinance path was smoke-
tested separately against SPY.

---

## What was built

### 1. Canonical schema — `meridian/data/schema.py`
`normalize_ohlcv(df)` coerces any raw frame into one fixed shape:
columns `["open", "high", "low", "close", "adj_close", "volume"]`, a
`DatetimeIndex` named `date`, sorted ascending. Missing required columns raise.
`OHLCV_COLUMNS` is the canonical order. No downstream module ever sees a
source's native column names.

### 2. Universe management — `meridian/data/universe.py`
- `Universe(name, symbols)` — frozen dataclass; a named symbol set.
- `get_universe(name, use_cache=True, cache=None)` resolves by name.
- **ETF universes** (`SPY`, `QQQ`, `IWM`) are static single-symbol sets.
- **Index universes** (`SP500`, `NASDAQ100`, `RUSSELL1000`) fetch constituents
  from Wikipedia (`pandas.read_html`) and cache them; yfinance has no
  constituent API. Tickers are normalized (`BRK.B` -> `BRK-B`).
- `KNOWN_UNIVERSES` lists all six names.

### 3. Caching — `meridian/data/cache.py`
- `OHLCVCache` — one **Parquet** file per `(symbol, interval)` under
  `data_cache/ohlcv/<interval>/<SYMBOL>.parquet`. `read` returns a normalized
  frame or `None` on miss; `write` normalizes before writing.
- `UniverseCache` — one **CSV** per index under `data_cache/universes/`.
- `data_cache/` is git-ignored. Caching is purely a speed concern: clearing it
  only re-triggers downloads, never changes results.

### 4. Ingestion — `meridian/data/loader.py`
- `load_ohlcv(symbol, start, end, interval="1d", use_cache=True, cache=None)`
  — the single entry point for prices. **On a miss it pulls full history
  (`period="max"`), caches it, then slices to `[start, end]`**, so overlapping
  requests reuse one file. yfinance's MultiIndex columns are flattened.
- `load_universe(symbols, ...)` -> `{symbol: frame}`, skipping symbols that
  fail so one bad ticker never aborts a whole pull.
- The raw download is isolated in `_download` so tests run fully offline.

### 5. Out-of-sample splitting — `meridian/data/splits.py`
- `SplitSpec` holds the three windows; defaults match CLAUDE.md
  (in-sample 2010-2019, OOS 2020-2022, walk-forward 2023+).
- `SplitSpec.from_config(data_cfg)` reads windows from an experiment config;
  `.validate()` enforces ordering and **non-overlap** (in-sample ends before
  OOS, OOS ends before walk-forward).
- `split(obj, spec=None)` partitions any DatetimeIndex-ed Series/DataFrame into
  `{"in_sample", "out_of_sample", "walk_forward"}`. This is the single guard
  against training on test data.

### 6. Tests — `tests/test_data.py`
15 offline tests: schema rename/validation, both caches' round-trip + miss,
loader download/cache/slice + cache-hit-skips-download + empty-raises +
universe skip-bad, ETF/unknown/index-fetch-then-cache + symbol normalization,
split partitioning + non-overlap + config parsing + overlap rejection.
**Full suite: 24 passed** (9 Phase 0 + 15 Phase 1).

**Live check:** `load_ohlcv("SPY", "2024-01-01", "2024-02-01")` returned 22
correctly-shaped bars from the real yfinance API.

---

## Interface contracts handed to later phases

```python
from meridian.data import (
    load_ohlcv,        # (symbol, start, end, interval="1d") -> OHLCV DataFrame
    load_universe,     # (symbols, ...) -> {symbol: DataFrame}
    get_universe,      # (name) -> Universe(name, symbols)
    split, SplitSpec,  # split(series_or_frame) -> {in_sample, out_of_sample, walk_forward}
    normalize_ohlcv, OHLCV_COLUMNS,
)
```

Canonical price frame: `DatetimeIndex` named `date`, columns
`open, high, low, close, adj_close, volume`. Estimators (Phase 2) should fit on
a price **Series** — typically the `close` or `adj_close` column.

---

## Known limitations / notes for Phase 2

- **Survivorship bias (unchanged, important).** Index constituents are fetched
  as of *today*, and yfinance omits delisted tickers — so any index-universe
  backtest is survivorship-biased on two counts. Must be documented in every
  report.
- **Wikipedia-sourced constituents** can change format; the fetcher scans all
  page tables for the expected symbol column and raises if none matches. Not
  yet pinned to a date/snapshot.
- **No corporate-action handling beyond `adj_close`.** We keep both raw `close`
  and `adj_close`; estimators should generally prefer `adj_close`.
- **yfinance pandas-3.0 ChainedAssignment FutureWarning** is emitted from
  inside yfinance (not our code); harmless for now.
- `load_universe` silently skips failures — fine for research pulls, but Phase 2
  experiment runners may want to surface which symbols were dropped.

---

## Next phase

**Phase 2 — Estimator library:** implement 37+ fair-value estimators against the
`BaseEstimator` contract, each unit-tested (`fit/update/predict_mean/residual/
zscore/state`) with `hypothesis` for mathematical properties. Estimators consume
a price `Series` from the Phase 1 loader.
