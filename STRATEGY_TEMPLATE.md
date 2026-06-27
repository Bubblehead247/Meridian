# STRATEGY TEMPLATE

Fill out every field before a strategy advances past the Research stage.

---

## Name

<!-- Unique, descriptive name for this strategy. -->

---

## Family

<!-- One of: mean_reversion / trend_following / momentum / breakouts /
     pullback_continuation / sector_rotation / long_term_etf /
     event_driven / volatility -->

---

## Model Type

<!-- Brief description of the core model mechanic (e.g., z-score reversion,
     RSI exhaustion, breakout of N-day range, etc.). -->

---

## Universe

<!-- Which ticker set this strategy trades (e.g., S&P 500, Nasdaq-100,
     Russell 1000, SPY/QQQ/IWM only, sector ETFs, etc.). -->

---

## Timeframe

<!-- Bar resolution used for signals (e.g., daily, 4-hour). -->

---

## Holding Period Bucket

<!-- One of: short (intraday–1d) / swing (2–10d) / intermediate (2–6wk) /
     long-term (1mo+) -->

---

## Entry Rules

<!-- Exact, unambiguous entry conditions. Reference indicator names,
     thresholds, and any required regime permission. -->

---

## Exit Rules

<!-- Exact exit conditions (target, time-based, signal reversal, etc.). -->

---

## Stop Rules

<!-- Hard stop and/or trailing stop logic. Specify in price distance,
     ATR multiples, or % of entry price. -->

---

## Position Sizing Rules

<!-- How size is calculated (e.g., fixed fractional, ATR-based, equal-weight).
     State the risk-per-trade assumption in % of equity. -->

---

## Regime Permission

<!-- Which (trend, volatility, breadth) regime combinations allow trading.
     Reference the permission matrix in PLAN.md §5. -->

---

## Cost Assumptions

<!-- Commission (bps), slippage (bps), and any borrow cost if short. -->

---

## Max Concurrent Positions

<!-- Maximum number of open positions at one time for this strategy. -->

---

## Max Capital Allocation

<!-- Maximum % of total portfolio equity this strategy may hold at once. -->

---

## Max Drawdown Limit Before Suspension

<!-- Strategy-level drawdown ceiling. Breach triggers automatic suspension
     and review. Express as % (e.g., 10%). -->

---

## Suspension Rule

<!-- What happens when the drawdown limit is breached: e.g., close all
     positions, set stage = suspended, notify monthly review. -->

---

## Current Graduation Stage

<!-- One of: research / backtest / walk_forward / oos / paper / pilot /
     proven / core / elite / retired -->

---

## Date Entered Current Stage

<!-- YYYY-MM-DD -->
