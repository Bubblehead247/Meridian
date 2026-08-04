"""Renders the monthly review data to a markdown report file under reports/.

This module owns rendering the monthly review output to markdown; it does NOT
own the evaluation logic (that stays in portfolio/monthly_review.py).
"""

from __future__ import annotations

from pathlib import Path

from meridian.portfolio.monthly_review import MonthlyReview, SleeveAction

DEFAULT_REPORT_DIR = Path("reports")


def _format_regime_summary(review: MonthlyReview) -> str:
    if review.regime is None:
        return "**Regime:** _not supplied_"
    trend, vol, breadth = review.regime
    return f"**Regime:** trend `{trend}` · volatility `{vol}` · breadth `{breadth}`"


def _format_portfolio_summary(review: MonthlyReview) -> str:
    heat = f"{review.portfolio_heat:.2%}" + (" ⚠️ breached" if review.heat_breached else "")
    ac = review.avg_correlation
    avg_corr = "n/a" if ac != ac else f"{ac:.2f}"   # NaN-safe
    lines = [
        f"- **Account equity:** {review.account_equity:,.0f}",
        f"- **Benchmark return:** {review.benchmark_return:.2%}",
        f"- **Portfolio heat:** {heat}",
        f"- **Avg inter-sleeve correlation:** {avg_corr}",
    ]
    if review.high_corr_pairs:
        pairs = ", ".join(f"{a}/{b}" for a, b in review.high_corr_pairs)
        lines.append(f"- **High-correlation pairs:** {pairs}")
    return "\n".join(lines)


def _sleeve_table(sleeves: list[SleeveAction], mctr: dict[str, float] | None = None) -> str:
    mctr = mctr or {}
    header = (
        "| Sleeve | Stage | Action | Return | vs Bench | DD | Risk Contrib | MCTR | Permitted |\n"
        "|---|---|---|---:|---:|---:|---:|---:|:---:|"
    )
    rows = [
        f"| {s.sleeve} | {s.stage} | **{s.action}** | {s.return_pct:.2%} | "
        f"{s.excess_vs_benchmark:+.2%} | {s.drawdown_cur:.1%} | "
        f"{s.risk_contribution:.0%} | {mctr.get(s.sleeve, 0.0):.0%} | "
        f"{'✅' if s.permitted else '⛔'} |"
        for s in sleeves
    ]
    note = (
        "\n\n_Risk Contrib = share of stop-distance heat. MCTR = share of "
        "portfolio volatility (correlation-adjusted) — a sleeve can be low on "
        "one and high on the other._"
    )
    return "\n".join([header, *rows]) + note


def _format_sleeve_section(sleeve: SleeveAction) -> str:
    notes = "\n".join(f"  - {r}" for r in sleeve.rationale) or "  - (within tolerances)"
    return f"### {sleeve.sleeve} → **{sleeve.action}**\n{notes}"


def render_monthly_report(review: MonthlyReview) -> str:
    """Render a `MonthlyReview` as a markdown string."""
    parts = [
        f"# Monthly Review — {review.as_of}",
        "",
        "## Portfolio",
        _format_portfolio_summary(review),
        "",
        _format_regime_summary(review),
        "",
        "## Sleeve actions",
        _sleeve_table(review.sleeves, review.marginal_risk_contribution),
        "",
        "## Rationale",
        *[_format_sleeve_section(s) for s in review.sleeves],
        "",
        "## Action summary",
    ]
    counts: dict[str, int] = {}
    for s in review.sleeves:
        counts[s.action] = counts.get(s.action, 0) + 1
    parts.append(" · ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return "\n".join(parts) + "\n"


def write_monthly_report(
    review: MonthlyReview, output_dir: str | Path = DEFAULT_REPORT_DIR
) -> Path:
    """Render and write the report to ``<output_dir>/monthly_review_<as_of>.md``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"monthly_review_{review.as_of}.md"
    path.write_text(render_monthly_report(review), encoding="utf-8")
    return path
