# Phase 9 Summary — Paper Trading Deployment

**Status:** Complete
**Date:** 2026-06-23

Phase 9 wires the validated pipeline to live execution in `meridian/execution/`.
The central guarantee: the live loop reuses the **exact same causal indicator
steps and signal state machine** as the backtester, so a paper session
reproduces a backtest's decisions bar for bar — no second, drifting
implementation of the trade logic.

---

## What was built

### 0. Refactor — shared `SignalState` (`signals/engine.py`)
Extracted the mean-reversion state machine from `generate_positions` into a
`SignalState` class with a `step(score) -> position` method. `generate_positions`
now just drives `SignalState` over a series; the live trader drives the **same
class** bar by bar. One implementation, used by both paths (all 18 signal tests
still pass unchanged).

### 1. Broker abstraction — `execution/broker.py`
- `BaseBroker` — interface the trader depends on: `get_position`, `market_order`.
- `SimulatedBroker` — deterministic in-memory paper broker: fills market orders
  at the last set price, applies flat-bps cost, tracks cash/positions, and
  marks-to-market (`equity()`). Used for tests and offline dry-runs.
- `AlpacaBroker` — thin alpaca-py wrapper (lazy import), credentials from args or
  `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`, **paper endpoint by default**. Submits
  market orders to reach a target; never called in tests.

### 2. Paper trader — `execution/trader.py`
`PaperTrader` holds an estimator, deviation metric, optional regime gate, a
`SignalState`, and a broker. Each bar (`on_bar`):
1. `estimator.update(price)` → `residual` → `deviation.update(resid, bar)` →
   `score` — the identical causal step `compute_scores` performs.
2. `SignalState.step(score)` → raw signal {-1,0,+1}.
3. Optional **regime gate** flattens the target when outside `allowed_regimes`
   (same masking semantics as `run_gated_backtest`).
4. Reconcile: order the difference between target and current broker position.

- `warm_up(prices, bars)` replays history through the indicators so they start
  live in the same state a backtest would have — the **book starts flat**.
- `replay(prices)` runs a full offline dry-run; `log_frame()` returns the
  decision log.

---

## Tests — `tests/test_execution.py`

10 tests (295 total in suite, all passing):
- **Simulated broker**: fills + cash/position accounting, mark-to-market equity,
  zero-order no-op, cost charging.
- **The key equivalence**: a `PaperTrader.replay` of a series produces signals
  **exactly equal** to `generate_positions(compute_scores(...))` — live
  reproduces backtest bar for bar.
- **Reconciliation**: broker position tracks the decided target; summed orders
  equal the net position; orders fire only on position changes; target =
  signal × size when ungated.
- **Regime gate** flattens the target on disallowed-regime bars.
- **Warm-up** primes the indicators (a warmed trader scores on its first live
  bar; a cold one returns NaN).
- **Alpaca guards**: missing credentials raise a clear error; `BaseBroker` is
  abstract.

**Live demonstration (SPY, ou + zscore, entry 1.5, 100-share size):** warmed on
300 bars, replayed 306 live bars through a `SimulatedBroker` — 46 orders, ended
flat at **$100,898** on $100k start, with the final decision correctly closing a
long when the score reverted above the exit level.

---

## Interface contracts handed to later phases

```python
from meridian.execution import PaperTrader, SimulatedBroker, AlpacaBroker
from meridian.signals import SignalConfig

broker = SimulatedBroker(cash=100_000, cost_bps=1.0)          # or AlpacaBroker(paper=True)
trader = PaperTrader("SPY", broker, estimator="ou", deviation="zscore",
                     signal=SignalConfig(entry_threshold=1.5), window=20,
                     position_size=100, regime=None, allowed_regimes=())
trader.warm_up(history_prices)
decision = trader.on_bar(latest_price, bar=latest_ohlc_row)   # live: call each bar
```

`estimator` accepts any registered name **including ensembles** (`ens_invvar`,
etc.) or an instance, so a validated meta-model deploys with no code change.

---

## Known limitations / notes for Phase 10

- **No live data feed / scheduler.** `on_bar` is the integration point; a
  Phase 10 runner must pull each new bar (alpaca-py data or yfinance) and call it
  on a schedule. `AlpacaBroker` is untested against the live API (needs creds).
- **Fills assumed at the bar price.** `SimulatedBroker` fills at the last set
  price with flat-bps cost; real fills have slippage and partial-fill risk.
- **Single symbol, fixed share size.** Portfolio-level sizing/allocation
  (`portfolio/`) and multi-symbol orchestration are not yet built (Phase 6→8 left
  `portfolio/` as a stub).
- **No persistence/restart.** Trader state lives in memory; a production deploy
  needs to checkpoint estimator/deviation/signal state to resume.
- **Survivorship and significance caveats unchanged** — going live does not make
  an un-validated edge real; deploy only configs that survived Phase 6.

---

## Next phase

**Phase 10 — Production packaging:** a config-driven runner (YAML → universe,
estimator, deviation, signal, regime, broker), scheduling/CLI entry points, state
persistence, and deployment docs — turning the library into a runnable service.
