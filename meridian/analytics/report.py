"""Markdown report generation for validated estimator rankings.

Builds a self-contained research report from a Phase 6 `validate()` table. The
**required disclosures are emitted automatically** — survivorship bias,
multiple-testing correction, out-of-sample discipline, and the flat-cost
assumption — so no report can accidentally omit them (a standing project rule).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from meridian.validation.external_benchmarks import HARVEY_LIU_ZHU_T_THRESHOLD

# The disclosures every report must carry (CLAUDE.md "Known Limitations").
_DISCLOSURES = """## Limitations and disclosures

- **Survivorship bias.** Price history comes from yfinance, which omits delisted
  companies, and index/screened universes are sampled as of today. Results are
  therefore survivorship-biased and likely optimistic. This bias is *not* removed
  by any statistic in this report.
- **Multiple testing.** Many estimators were tested on shared data, so some will
  look good by chance. Significance is reported only after correction; raw
  p-values must not be read on their own.
- **Out-of-sample discipline.** Reported performance is walk-forward
  out-of-sample. In-sample figures (if any) are sanity checks, not results.
- **Transaction costs** are modeled as flat basis points on turnover; real
  slippage, spread, impact, and borrow costs will differ.
- **Short mechanics.** Short positions in this backtest are frictionless: no
  borrow cost, no dividends owed, no hard-to-borrow/locate constraints, and no
  asymmetric execution vs. longs. Separately, the live/paper execution path
  (``execution/live_runner.py``) currently flattens every short signal to no
  position — a short-capable strategy's backtested P&L therefore includes
  trades that would never actually be taken live under the current execution
  code, not just trades taken at an optimistic cost. Do not treat a backtest
  Sharpe as achievable live for any strategy whose signal path takes short
  positions until this is resolved.
"""


def _fmt(x) -> str:
    if isinstance(x, float):
        if x != x:  # NaN
            return "—"
        return f"{x:.3f}"
    return str(x)


def _table_md(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    cols = columns or list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_validation_report(
    table: pd.DataFrame,
    meta: dict | None = None,
    *,
    title: str = "Meridian — Mean-Reversion Estimator Validation",
) -> str:
    """Render a markdown report from a `validate()` results table.

    Args:
        table: The ranked DataFrame from `meridian.validation.validate`.
        meta: Optional run metadata (universe, symbol, date range, config,
            bootstrap/MC settings) shown in a setup section.
        title: Report title.

    Returns:
        The full report as a markdown string.
    """
    meta = meta or {}
    sig = table[table["significant"]] if "significant" in table.columns else table.iloc[0:0]

    parts: list[str] = [f"# {title}", "", f"_Generated {date.today().isoformat()}_", ""]

    # Survivorship-bias banner — dynamic, not just the boilerplate disclosure below,
    # so a reader can't miss which universe-resolution path actually produced this run.
    if "survivorship_biased" in meta:
        if meta["survivorship_biased"]:
            parts.append(
                "> **⚠ Survivorship-biased run.** This universe was resolved from "
                "current constituents/available tickers, not a point-in-time "
                "membership list. Results are likely optimistic; do not treat this "
                "run alone as evidence of edge."
            )
        else:
            parts.append(
                "> **✓ Point-in-time universe.** This run used the survivorship-free "
                "constituent dataset (`data/survivorship.py`) — membership is gated "
                "to each date, including names later delisted."
            )
            if "survivorship_coverage_warning" in meta:
                parts.append(
                    f"> **⚠ Coverage gap:** the dataset's actual price coverage is "
                    f"{meta['survivorship_coverage_warning']} Do not read this run as "
                    "validating the full requested window — only the covered subset "
                    "was actually tested."
                )
        parts.append("")

    # Setup / configuration
    if meta:
        parts.append("## Setup")
        parts.append("")
        for k in ("universe", "symbols", "start", "end", "deviation", "window",
                  "entry_threshold", "cost_bps", "wfo", "n_boot", "n_mc", "correction"):
            if k in meta:
                parts.append(f"- **{k}**: {meta[k]}")
        parts.append("")

    # Headline verdict
    parts.append("## Verdict")
    parts.append("")
    if len(sig):
        names = ", ".join(f"`{e}`" for e in sig["estimator"])
        parts.append(
            f"**{len(sig)} of {len(table)} estimators are statistically significant** "
            f"after multiple-testing correction: {names}."
        )
    else:
        parts.append(
            f"**No estimator is statistically significant** after multiple-testing "
            f"correction (of {len(table)} tested). The best out-of-sample Sharpe is "
            f"`{table.iloc[0]['estimator']}` at {_fmt(table.iloc[0]['oos_sharpe'])}, but its "
            f"corrected q-value does not clear the threshold — consistent with no "
            f"robust mean-reversion edge on this data."
        )
    parts.append("")

    # Ranked results
    parts.append("## Ranked results (out-of-sample)")
    parts.append("")
    parts.append(_table_md(table))
    parts.append("")

    _append_robustness_section(parts, table, meta)

    parts.append(_DISCLOSURES)
    return "\n".join(parts)


def _append_robustness_section(parts: list[str], table: pd.DataFrame, meta: dict) -> None:
    """Phase 2 additions: collinearity-adjusted correction, DSR, sensitivity.

    Purely additive to the report — never changes the headline ``## Verdict``
    section above, which stays keyed to the raw-m ``significant`` column.
    """
    has_eff = "m_eff" in table.columns and "significant_eff" in table.columns
    has_dsr = "dsr_pvalue" in table.columns
    has_sensitivity = "sharpe_sign_stable" in table.columns
    has_cumulative = "cumulative_trial_count" in meta
    has_pbo = "pbo" in table.columns
    has_hlz = "t_stat_classical" in table.columns
    if not (has_eff or has_dsr or has_sensitivity or has_cumulative or has_pbo or has_hlz):
        return

    parts.append("## Robustness")
    parts.append("")

    if has_cumulative:
        cum = meta["cumulative_trial_count"]
        m_raw = len(table)
        parts.append(
            f"- **Cumulative research history.** `q_value`/`dsr_pvalue` above account "
            f"only for the `m={m_raw}` estimators tested *in this run*. Across every "
            f"logged run against this same symbol/universe scope "
            f"(`reports/run_log.jsonl`), **{cum} distinct (estimator, deviation, "
            f"window) trials have been tried in total.** If {cum} > {m_raw}, this run "
            f"is not a fresh, unpenalized look at the data — treat the correction "
            f"above as understating the true multiple-testing burden."
        )
        parts.append("")

    if has_eff:
        m_eff = table["m_eff"].iloc[0]
        m_raw = len(table)
        flipped = table[table["significant"] != table["significant_eff"]]
        parts.append(
            f"- **Collinearity-adjusted correction.** {m_raw} estimators were tested, "
            f"but their OOS returns are correlated — the effective independent test "
            f"count is `m_eff={_fmt(m_eff)}`. The headline verdict above uses the raw "
            f"(conservative) `m={m_raw}` correction."
        )
        if len(flipped):
            names = ", ".join(f"`{e}`" for e in flipped["estimator"])
            parts.append(
                f"  ⚠ **{len(flipped)} estimator(s) only clear the bar under the "
                f"effective-m correction, not the raw one:** {names}. Treat these as "
                f"a weaker, collinearity-adjusted signal, not the headline result."
            )
        else:
            parts.append("  No estimator's significance flips between the raw and effective correction.")
        parts.append("")

    if has_dsr:
        best = table.iloc[0]
        parts.append(
            f"- **Deflated Sharpe Ratio** (best performer, `{best['estimator']}`): "
            f"`dsr_pvalue={_fmt(best['dsr_pvalue'])}` — probability the true Sharpe "
            f"exceeds the expected best-of-N maximum under zero skill, given how many "
            f"estimators were tried. Unlike a normal p-value, *higher* is more "
            f"significant (commonly read as significant above ~0.95)."
        )
        if "dsr_pvalue_eff" in table.columns:
            parts.append(
                f"  Using the collinearity-adjusted trial count (`m_eff`) instead of "
                f"the raw estimator count: `dsr_pvalue_eff={_fmt(best['dsr_pvalue_eff'])}`. "
                f"These can disagree — `dsr_pvalue` (raw m) is the conservative, "
                f"backward-compatible default; `dsr_pvalue_eff` is the internally-"
                f"consistent one (same trial count as `q_value_eff` above). Neither is "
                f"cumulative across separate validation runs against this data — see "
                f"the run log for that."
            )
        parts.append("")

    if has_hlz:
        best = table.iloc[0]
        n_clear = int(table["significant_hlz"].sum())
        parts.append(
            f"- **External benchmark (Harvey-Liu-Zhu).** {n_clear} of {len(table)} "
            f"estimator(s) clear the classical `|t| > {HARVEY_LIU_ZHU_T_THRESHOLD:.1f}` "
            f"hurdle Harvey, Liu & Zhu (2016) recommend for a *newly proposed* factor, "
            f"given how much data-mining has occurred across the finance literature as "
            f"a whole — a stricter, independently-sourced skepticism check, not a "
            f"fourth correction stacked on top of the ones above. Best performer "
            f"`{best['estimator']}`: `t_stat_classical={_fmt(best['t_stat_classical'])}`. "
            f"This t-statistic does not correct for serial correlation (unlike the "
            f"block bootstrap above) or non-normality (unlike DSR) — it is "
            f"deliberately the crude textbook version the 3.0 hurdle is calibrated "
            f"against."
        )
        parts.append("")

    if has_pbo:
        pbo = table["pbo"].iloc[0]
        parts.append(
            f"- **Probability of Backtest Overfitting (CPCV/PBO)**: `pbo={_fmt(pbo)}`. "
            f"Across many resampled train/test partitions of this same data, this is "
            f"how often the in-sample-best estimator ranked at or below the "
            f"out-of-sample median — i.e. how often picking 'the winner' would have "
            f"picked noise. Complements (does not replace) walk-forward/bootstrap/DSR "
            f"above; the exact formula's numerical output is not independently "
            f"verified against its primary source — see `validation/cpcv.py`'s "
            f"docstring for exactly what is and isn't confirmed."
        )
        parts.append("")

    if has_sensitivity:
        unstable = table[~table["sharpe_sign_stable"]]
        if len(unstable):
            names = ", ".join(f"`{e}`" for e in unstable["estimator"])
            parts.append(
                f"- ⚠ **Parameter sensitivity:** {len(unstable)} estimator(s) flip sign "
                f"across neighboring windows (not just the fixed one used for ranking): "
                f"{names}. Treat their result as a knife-edge artifact of the exact "
                f"window chosen, not a robust finding."
            )
        else:
            parts.append(
                "- **Parameter sensitivity:** every estimator's OOS Sharpe sign is "
                "stable across neighboring windows."
            )
        parts.append("")


def write_report(path: str | Path, markdown: str) -> Path:
    """Write a report string to disk, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
