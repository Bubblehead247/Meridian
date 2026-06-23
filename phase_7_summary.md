# Phase 7 Summary — Analytics and Reporting

**Status:** Complete
**Date:** 2026-06-22

Phase 7 delivers the presentation layer: full performance metrics
(`meridian/analytics/`), comparison charts (`meridian/visualization/`), and a
markdown report generator that renders the Phase 6 verdict with the **mandatory
disclosures baked in**. This is how validated rankings become a shareable
research report.

---

## What was built

### 1. Performance metrics — `meridian/analytics/metrics.py`
`performance_metrics(returns, trades=None)` → a full dict from a per-bar return
series: total/CAGR return, annualized vol, **Sharpe / Sortino / Calmar**, max
drawdown + drawdown duration, hit rate, profit factor, avg win/loss, skew,
kurtosis, VaR/CVaR (95%), best/worst bar, tail ratio. With an optional trade
ledger it adds trade win-rate, expectancy, and trade profit factor. Plus
`equity_curve`, `drawdown_series`, `max_drawdown`, `max_drawdown_duration`.
All NaN-safe; Phase 6 used one Sharpe, this is the descriptive layer.

### 2. Report generator — `meridian/analytics/report.py`
- `build_validation_report(table, meta)` → a self-contained markdown report from
  a `validate()` table: setup section, a **plain-language verdict** (names the
  winners, or states none are significant and why), the ranked OOS results
  table, and the disclosures.
- **The required disclosures are emitted automatically** — survivorship bias,
  multiple-testing, out-of-sample discipline, flat-cost assumption — so no
  report can omit them (a standing project rule, now enforced in code).
- `write_report(path, md)` writes to disk (creating dirs).

### 3. Charts — `meridian/visualization/charts.py`
Matplotlib figures (headless **Agg** backend, server/CI-safe):
- `sharpe_heatmap(table)` — estimator × deviation Sharpe grid, diverging colors.
- `equity_curves({name: returns})` — overlaid OOS equity curves.
- `drawdown_plot(returns)` — filled underwater plot.
- `save_fig(fig, path)` — write a PNG and close the figure.

---

## Tests — `tests/test_analytics.py`, `tests/test_visualization.py`

28 tests (261 total in suite, all passing):
- **Metrics**: exact drawdown and max-drawdown, underwater-duration counting,
  all expected fields present, hit-rate bounds, all-wins → infinite profit
  factor, empty input safe, trade-level stats from a ledger.
- **Report**: all four mandatory disclosures present, "no significant" verdict
  names the best performer, "significant" verdict lists winners, `write_report`
  creates the file.
- **Charts**: each function returns a matplotlib `Figure`; `save_fig` writes a
  non-empty PNG (Agg backend, no display needed).

**Live artifact (SPY 2010-2024, 8 estimators):** generated
`reports/spy_validation.md` and `reports/spy_sharpe_heatmap.png`. The report's
auto-verdict: *"No estimator is statistically significant … best is `hull` at
0.413, but its corrected q-value does not clear the threshold — consistent with
no robust mean-reversion edge on this data."* (`reports/` is the output dir and
is git-ignored.)

---

## Interface contracts handed to later phases

```python
from meridian.analytics import performance_metrics, build_validation_report, write_report
from meridian.visualization import sharpe_heatmap, equity_curves, save_fig

m  = performance_metrics(result.returns, trades=result.trades)   # full metrics dict
md = build_validation_report(validate_table, meta)               # markdown (auto-disclosures)
write_report("reports/run.md", md)
save_fig(sharpe_heatmap(wf.summary()), "reports/heatmap.png")
```

---

## Known limitations / notes for Phase 8+

- **Reports are markdown.** PDF export (pyproject lists no PDF engine yet) can be
  added via pandoc/weasyprint in Phase 10 packaging if needed.
- **Metrics assume per-bar returns at one frequency** (daily default). Mixed
  frequencies would need re-annualization.
- **Charts are static matplotlib.** Interactive plotly dashboards (plotly is a
  dependency) are a possible Phase 8+ addition; not required for the comparison.
- **No PDF/tear-sheet of an individual estimator yet** — the report is the
  cross-estimator ranking. Per-estimator tear sheets are a straightforward
  extension using the same metrics + charts.
- **Survivorship and multiple-testing caveats are textual**, emitted every
  report; they describe but do not correct the underlying data bias.

---

## Next phase

**Phase 8 — Adaptive meta-model and ensemble:** combine estimators (e.g. the
walk-forward selection overlay from Phase 6, or a weighted ensemble of
fair-value estimates) into an adaptive model, validated through the same Phase 6
engine and reported through Phase 7.
