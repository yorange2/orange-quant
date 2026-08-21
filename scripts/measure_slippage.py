"""实盘真实交易成本：滑点 + 手续费 vs 回测假设的 cost_rate。

回测按 `backtest.cost_rate`（现为 0.1%）单边计费，并且假定成交价**精确等于**
决策那根 bar 的 close。实盘两条都不成立：

  * 手续费是 Binance 实收（taker 0.1%，无 BNB 折扣时与回测一致）；
  * 成交发生在 bar 收盘后几分钟，且吃的是盘口另一侧 —— 这部分回测完全没算。

滑点定义（正 = 不利 = 成本）::

    买入   +(fill − close) / close
    卖出   −(fill − close) / close

`close` 取 `latest_closed_bar(成交时刻)` 那根，即 `(ts − 1h).floor('h')` ——
与 `Rolling24hScorer` 判定"最新已收盘 bar"的规则完全一致，所以对比的是模型
真正看到的那个价格。

成交记录从 Binance `fetchMyTrades` 拉取（不依赖 state 文件——它每轮被覆盖），
因此可反复运行，样本随交易轮次自然累积。

用法（在 orange-quant/ 下）::

    ../.venv/bin/python scripts/measure_slippage.py --days 7
    ../.venv/bin/python scripts/measure_slippage.py --days 30 --by-round
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from orange_quant.lgb.dataset import load_config
from orange_quant.rl.dataset import bar_reader
from orange_quant.trading import make_broker


def decision_close(bars: pd.DataFrame, ts: pd.Timestamp):
    """成交时刻 ts 所对应的决策 bar 收盘价（模型当时看到的那个 close）。"""
    anchor = (ts - pd.Timedelta(1, "h")).floor("h")
    if bars is None or anchor not in bars.index:
        return None, anchor
    return float(bars.loc[anchor, "close"]), anchor


def main() -> None:
    ap = argparse.ArgumentParser(description="实盘滑点与真实交易成本")
    ap.add_argument("--config", default="binance-lgb-rolling24h-lag0f1")
    ap.add_argument("--days", type=int, default=7, help="回溯天数")
    ap.add_argument("--by-round", action="store_true", help="额外按轮次(日期)汇总")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cost_rate_bps = float(cfg["backtest"]["cost_rate"]) * 1e4
    codes = list(cfg["universe"]["codes"])
    quote = cfg["market"].get("quote_ccy", "USDT")

    broker = make_broker(cfg, "binance")
    ex = broker.exchange
    read = bar_reader({**cfg, "data": {k: v for k, v in cfg["data"].items()
                                       if k != "hour_of_day"}})
    since = int((pd.Timestamp.utcnow() - pd.Timedelta(args.days, "D")).timestamp() * 1000)

    rows = []
    for coin in codes:
        try:
            trades = ex.fetch_my_trades(f"{coin}/{quote}", since=since, limit=1000)
        except Exception:
            continue                      # 已下架/无此交易对
        if not trades:
            continue
        bars = read(coin)
        for t in trades:
            ts = pd.Timestamp(t["datetime"]).tz_localize(None)
            close, anchor = decision_close(bars, ts)
            if close is None or not t.get("cost"):
                continue
            sign = 1.0 if t["side"] == "buy" else -1.0
            slip = sign * (t["price"] - close) / close * 1e4
            # 手续费币种不固定：买入时 Binance 默认从**买到的币**里扣，卖出时扣
            # USDT。直接拿 fee.cost 除以 USDT 名义额会把币的数量当成金额，
            # 算出 BICO 296bp 这种离谱费率。必须先按成交价折成计价货币。
            fee = np.nan
            if t.get("fee") and t["fee"].get("cost") is not None:
                fee_ccy, fee_amt = t["fee"]["currency"], t["fee"]["cost"]
                if fee_ccy == quote:
                    fee_quote = fee_amt
                elif fee_ccy == coin:
                    fee_quote = fee_amt * t["price"]
                else:
                    fee_quote = None          # 例如 BNB 抵扣，需另取 BNB 价，跳过
                if fee_quote is not None:
                    fee = fee_quote / t["cost"] * 1e4
            rows.append({"coin": coin, "date": ts.strftime("%Y-%m-%d"), "ts": ts,
                         "side": t["side"], "notional": t["cost"], "fill": t["price"],
                         "close": close, "anchor": anchor,
                         "slip_bps": slip, "fee_bps": fee})

    if not rows:
        print("没有可用成交记录")
        return
    df = pd.DataFrame(rows)

    # 同一订单常被拆成多笔 fill，按 (轮次, 币, 方向) 以名义额加权合并成一次决策
    def agg(g):
        w = g.notional / g.notional.sum()
        return pd.Series({"notional": g.notional.sum(),
                          "slip_bps": float((g.slip_bps * w).sum()),
                          "fee_bps": float((g.fee_bps * w).sum()),
                          "n_fills": len(g)})
    o = df.groupby(["date", "coin", "side"]).apply(agg, include_groups=False).reset_index()

    def wavg(g, col):
        return float((g[col] * g.notional).sum() / g.notional.sum())

    print("=" * 78)
    print(f"实盘交易成本  |  回溯 {args.days} 天  |  回测假设 cost_rate = {cost_rate_bps:.1f} bp/单边")
    print("=" * 78)
    print(f"  订单数 {len(o)}（{int(df.groupby(['date','coin','side']).size().sum())} 笔 fill）"
          f"  总名义额 {o.notional.sum():,.2f} {quote}  轮次 {o.date.nunique()}")

    print(f"\n{'币':<8}{'单数':>5}{'名义额':>11}{'滑点bp':>10}{'手续费bp':>10}{'合计bp':>9}"
          f"{'vs回测':>10}")
    per = []
    for coin, g in o.groupby("coin"):
        s, f = wavg(g, "slip_bps"), wavg(g, "fee_bps")
        per.append((s + f, coin, len(g), g.notional.sum(), s, f))
    for tot, coin, n, notional, s, f in sorted(per, reverse=True):
        print(f"{coin:<8}{n:>5}{notional:>11.2f}{s:>+10.2f}{f:>10.2f}{tot:>+9.2f}"
              f"{tot - cost_rate_bps:>+10.2f}")

    S, F = wavg(o, "slip_bps"), wavg(o, "fee_bps")
    print(f"\n{'合计':<8}{len(o):>5}{o.notional.sum():>11.2f}{S:>+10.2f}{F:>10.2f}"
          f"{S+F:>+9.2f}{S+F-cost_rate_bps:>+10.2f}")
    # 买卖对称性分解。设成交时刻 mid = close×(1+d)、半价差 h，则
    #   买入滑点 = d + h，卖出滑点 = −d + h
    # → h =（买+卖）/2 是真正付出去的价差成本（永远不利，系统性）；
    #   d =（买−卖）/2 是 bar 收盘到成交这几分钟的市场漂移（随机，长期应趋 0）。
    # 不拆开的话，一段单边行情会被整个误读成滑点。
    buy = wavg(o[o.side == "buy"], "slip_bps")
    sell = wavg(o[o.side == "sell"], "slip_bps")
    h, d = (buy + sell) / 2, (buy - sell) / 2
    print(f"\n  按方向：买入滑点 {buy:+.2f} bp，卖出滑点 {sell:+.2f} bp")
    print(f"  分解：  半价差 h = {h:+.2f} bp（系统性成本）　"
          f"市场漂移 d = {d:+.2f} bp（随机，长期应趋 0）")

    real = h * 2 + F      # 单边真实成本 = 双边价差 + 手续费；漂移不计入系统性成本
    print(f"\n  手续费 {F:.2f} bp + 价差 {h*2:.2f} bp = 真实单边成本 {real:.2f} bp"
          f"  vs 回测 {cost_rate_bps:.1f} bp → {real - cost_rate_bps:+.2f} bp"
          f"（{real / cost_rate_bps:.2f}×）")
    turnover = 26.2       # ROADMAP 记录的 topk=10 年换手预测，实测够样本后应替换
    print(f"  按年换手 {turnover:.1f} 估算年成本：回测 {cost_rate_bps*turnover/1e4*100:.2f}%"
          f" → 实际 {real*turnover/1e4*100:.2f}%")

    if args.by_round:
        print(f"\n{'轮次':<12}{'单数':>5}{'名义额':>11}{'滑点bp':>10}{'手续费bp':>10}{'合计bp':>9}")
        for d, g in o.groupby("date"):
            s, f = wavg(g, "slip_bps"), wavg(g, "fee_bps")
            print(f"{d:<12}{len(g):>5}{g.notional.sum():>11.2f}{s:>+10.2f}{f:>10.2f}{s+f:>+9.2f}")


if __name__ == "__main__":
    main()
