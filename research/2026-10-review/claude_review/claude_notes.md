# Notes kept out of the review (it had to use packet facts only)

**Errors in the packet (Claude wrote it):**
- §3 says "the idle 40% came from the single-stock sleeves". It was 32.5% single-stock sleeves
  plus the 5% cash reserve plus SVXY's 5% when flat.
- §5 test 2 and §4b omit that the lab was patched between committing the structure plan and
  running it (the first run crashed before printing a result; candidates and rule unchanged).
- §3 omits the live-window check: a backtest of the live picks gave +0.27% against the account's
  −0.03% over 6/29–9/25.

**Facts outside the packet that bear on the findings:**
- F5 (monitoring) verified in code: a rejected order only prints `order failed` to `meridian.log`
  (`live_runner.py`); `notify.py` pushes entries, exits, fills and a daily status, not failures.
- Infrastructure (a missed risk in the review): all three bots run on one PC. A Windows Update
  reboot on 2026-09-15 crashed all three tray apps for 12 days; the scheduled jobs survived.
- The execution cost of SPY/IEF/SGOV at the open is small in general knowledge, but it hasn't
  been measured here; the −3.9 bps "slippage" is overnight gap plus cost, and is mostly gap.
