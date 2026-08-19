"""Reconcile a live run: what the strategy intended vs what the venue actually did.

The runner's state file records *intent* (target weights) and the broker's
response per order; the exchange records the truth. They diverge for reasons
that are all invisible in the logs alone:

  * ``amount_to_precision`` truncates downward, so a $10 buy can fill $9.64 —
    the coarser the lot step and the smaller the order, the bigger the gap;
  * fees are charged in the *base* asset on buys, so the coin received is less
    than the coin bought;
  * ``fit_buys_to_cash`` may have scaled or dropped buys when sells funded less
    than planned;
  * an order can come back with ``error`` and simply not exist.

So this compares three things side by side: target weight, the orders the
broker accepted, and the position actually held now. Anything that does not
line up is printed as a finding rather than left for eyeball diffing.

Run from orange-quant/::
    ../.venv/bin/python scripts/reconcile_live.py --config binance-lgb-rolling24h
    ../.venv/bin/python scripts/reconcile_live.py --config ... --container orange-quant-rolling24h
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from orange_quant.rl.dataset import load_config  # noqa: E402
from orange_quant.trading import make_broker  # noqa: E402

LOG_KEYS = ("live universe", "pruned", "rolling24h:", "STALE", "scaling",
            "drop buy", "orders filled", "firing for", "run result",
            "not trading", "failed")


def show_logs(container: str, lines: int) -> None:
    print("=" * 78)
    print(f"1. 容器日志 ({container})")
    print("=" * 78)
    try:
        out = subprocess.run(["docker", "logs", "--tail", str(lines), container],
                             capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001 - reconciliation must not die on docker
        print(f"  无法读取日志: {e}")
        return
    blob = (out.stdout or "") + (out.stderr or "")
    hits = [l for l in blob.splitlines() if any(k in l for k in LOG_KEYS)]
    print("\n".join(f"  {l}" for l in hits[-40:]) if hits
          else "  (没有匹配的关键行——这一轮可能还没跑)")


def show_state(state_file: Path) -> dict:
    print("\n" + "=" * 78)
    print(f"2. 运行记录 ({state_file})")
    print("=" * 78)
    if not state_file.exists():
        print("  state 文件不存在——这一轮没跑过")
        return {}
    s = json.loads(state_file.read_text())
    print(f"  date={s.get('date')}  skipped={s.get('skipped', False)}  "
          f"portfolio_value={s.get('portfolio_value')}")
    for k in ("blend", "views_used", "data_window", "max_lag_hours", "stale",
              "coins_scored", "pruned_universe"):
        if k in s:
            print(f"  {k:18} {s[k]}")

    orders = s.get("orders") or []
    ok = [o for o in orders if o.get("result") is not None]
    bad = [o for o in orders if o.get("error")]
    print(f"\n  订单 {len(ok)}/{len(orders)} 成交")
    for o in bad:
        print(f"    ✗ {o['side']:<4} {o['coin']:<10} {o.get('error')}")
    return s


def reconcile(cfg: dict, state: dict) -> None:
    print("\n" + "=" * 78)
    print("3. 目标 vs 实际持仓")
    print("=" * 78)
    targets = state.get("targets") or {}
    if not targets:
        print("  无目标权重，跳过对账")
        return

    broker = make_broker(cfg, "binance")
    quote = cfg["market"]["quote_ccy"]
    bal = broker.get_balances() or {}
    coins = sorted(set(targets) | {c for c, v in bal.items()
                                   if c != quote and v > 0})
    prices = broker.get_current_prices(coins) or {}
    cash = float(bal.get(quote, 0.0))
    held = {c: float(bal.get(c, 0.0)) * (prices.get(c) or 0.0) for c in coins}
    equity = cash + sum(held.values())
    if equity <= 0:
        print("  账户权益为 0，跳过")
        return

    print(f"  {quote} 现金 {cash:,.2f}   持仓 {sum(held.values()):,.2f}   "
          f"总权益 {equity:,.2f}")

    # A paper run's state file records PaperBroker's fictional $100k account. If
    # that ever lands in the live path the weights below are computed against a
    # portfolio that does not exist, so say so instead of printing a table of
    # meaningless deviations.
    recorded = float(state.get("portfolio_value") or 0.0)
    if recorded > 0 and not (0.5 < recorded / equity < 2.0):
        print(f"\n  ⚠ state 记录的组合价值 {recorded:,.2f} 与账户实际权益 "
              f"{equity:,.2f} 差 {recorded / equity:.1f}×")
        print("    —— 这条记录很可能来自纸面运行，下面的偏差不代表实盘执行结果")
    print(f"\n  {'币':<10}{'目标权重':>10}{'实际权重':>10}{'偏差':>9}"
          f"{'实际名义':>12}")
    findings = []
    for c in sorted(coins, key=lambda x: -held.get(x, 0)):
        tw = targets.get(c, 0.0)
        aw = held.get(c, 0.0) / equity
        if held.get(c, 0.0) < 1.0 and tw == 0:
            continue                                  # dust, never targeted
        print(f"  {c:<10}{tw:>10.4f}{aw:>10.4f}{aw - tw:>+9.4f}"
              f"{held.get(c, 0):>12.2f}")
        if tw > 0 and held.get(c, 0.0) < 1.0:
            findings.append(f"{c}: 目标 {tw:.4f} 但几乎没有持仓（买单可能失败/被丢弃）")
        if tw == 0 and held.get(c, 0.0) >= 20:
            findings.append(f"{c}: 非目标却仍持有 {held[c]:.2f} {quote}（卖单可能失败）")

    deployed = sum(held.get(c, 0) for c in targets) / equity
    want = sum(targets.values())
    print(f"\n  目标部署比例 {want:.4f}   实际 {deployed:.4f}   "
          f"差 {deployed - want:+.4f}")

    print("\n" + "=" * 78)
    print("4. 发现")
    print("=" * 78)
    if findings:
        for f in findings:
            print(f"  ⚠ {f}")
    else:
        print("  ✓ 目标与持仓一致，无异常")


def main() -> None:
    ap = argparse.ArgumentParser(description="实盘运行对账")
    ap.add_argument("--config", default="binance-lgb-rolling24h")
    ap.add_argument("--container", default="orange-quant-rolling24h")
    ap.add_argument("--lines", type=int, default=400)
    args = ap.parse_args()

    cfg = load_config(args.config)
    state_file = Path(cfg.get("trading", {}).get(
        "state_file", "data/live_state/state.json"))

    show_logs(args.container, args.lines)
    state = show_state(state_file)
    if state and not state.get("skipped"):
        reconcile(cfg, state)
    else:
        print("\n(这一轮被跳过或未执行，不做持仓对账)")


if __name__ == "__main__":
    main()
