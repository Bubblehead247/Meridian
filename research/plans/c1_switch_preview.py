"""Preview the orders the C1 switch would send, from the account's real holdings.

Read-only and offline to the broker:
  * holdings and equity are READ from Meridian's Alpaca paper account;
  * the live books (positions.json, last_rebalance.json) are COPIED into a
    temporary folder, which is also the working directory, so nothing the
    session writes can reach the live ledger;
  * orders go to SimulatedBroker (fills at the last close, no cost) — nothing
    is sent to Alpaca; notifications are off (no NTFY_TOPIC).

    python research/plans/c1_switch_preview.py   (from the C1 worktree)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

LIVE = Path(r"C:\Users\Cody\Claude\Meridian")
REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    os.environ.pop("NTFY_TOPIC", None)
    sys.path.insert(0, str(REPO))
    from dotenv import dotenv_values
    from alpaca.trading.client import TradingClient

    keys = dotenv_values(LIVE / ".env")
    client = TradingClient(keys["ALPACA_API_KEY"], keys["ALPACA_SECRET_KEY"], paper=True)
    acct = client.get_account()
    broker_pos = client.get_all_positions()
    held = {p.symbol: float(p.qty) for p in broker_pos}
    mark = {p.symbol: float(p.current_price) for p in broker_pos}
    equity = float(acct.equity)

    tmp = Path(tempfile.mkdtemp(prefix="c1_preview_"))
    (tmp / "ledger").mkdir()
    for name in ("positions.json", "last_rebalance.json"):
        if (LIVE / "ledger" / name).exists():
            shutil.copy(LIVE / "ledger" / name, tmp / "ledger" / name)
    os.chdir(tmp)

    from meridian.execution import live_runner
    from meridian.execution.broker import SimulatedBroker

    broker = SimulatedBroker(cash=float(acct.cash), cost_bps=0.0)
    broker.positions.update(held)
    sent: list[tuple[str, float, float]] = []
    orig = broker.market_order

    def record(symbol, qty, **kw):
        fill = orig(symbol, qty, **kw)
        sent.append((symbol, qty, broker.last_price.get(symbol, 0.0)))
        return fill

    broker.market_order = record
    live_runner.run_paper_session(
        broker, account_equity=equity,
        positions_path=tmp / "ledger" / "positions.json",
        rebalance_schedule_path=tmp / "ledger" / "last_rebalance.json",
    )

    print(f"\nC1 switch preview as of {date.today()} - account equity ${equity:,.2f} "
          f"(read-only; simulated fills; books copied to {tmp})\n")
    print("| Symbol | Order (shares) | ~$ | Held before | Held after |")
    print("|---|---:|---:|---:|---:|")
    px = {**mark, **broker.last_price}
    for sym, qty, _ in sorted(sent, key=lambda x: x[0]):
        p = px.get(sym, 0.0)
        print(f"| {sym} | {qty:+.4f} | {qty * p:+,.0f} | {held.get(sym, 0.0):.4f} | {broker.positions.get(sym, 0.0):.4f} |")
    after = {s: q for s, q in broker.positions.items() if abs(q) > 1e-6}
    value = {s: q * px.get(s, 0.0) for s, q in after.items()}
    total = sum(value.values())
    print(f"\nAfter the switch: invested ${total:,.0f} of ${equity:,.0f} "
          f"({total / equity:.0%}); simulated cash ${broker.cash:,.0f}")
    for s, v in sorted(value.items(), key=lambda x: -x[1]):
        print(f"  {s:5} ${v:>9,.0f}  {v / equity:5.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
