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

    parts.append(_DISCLOSURES)
    return "\n".join(parts)


def write_report(path: str | Path, markdown: str) -> Path:
    """Write a report string to disk, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
