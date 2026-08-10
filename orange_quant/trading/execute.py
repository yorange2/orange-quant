"""Shared diff-and-execute portfolio rebalancing (RL rotation + LGB rotation).

Extracted from ``orange_quant.live.RLRotationRunner._execute`` so the two
runners place orders identically: diff target weights against current
holdings, skip dust-sized deltas, drop buys of reduce-only (blacklisted)
coins, place market orders with per-order try/except isolation.

The blacklist is applied here rather than in one runner so both strategies
honour it, and the path comes from the broker that writes it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set


def diff_orders(target_w: Dict[str, float], codes: List[str], balances: Dict[str, float],
                prices: Dict[str, float], quote_ccy: str,
                min_notional_safety: float,
                blacklist: Optional[Set[str]] = None) -> tuple:
    """Build order list from target weights vs holdings. Returns (orders, value).

    Blacklisted (reduce-only) coins are never bought; selling them stays
    allowed so an existing position can still be exited."""
    blacklist = blacklist or set()
    value = float(balances.get(quote_ccy, 0.0)) + sum(
        float(balances.get(c, 0.0)) * prices.get(c, 0.0) for c in codes)
    if value <= 0:
        return [], value

    orders: List[dict] = []
    for coin in codes:
        tgt = target_w.get(coin, 0.0) * value
        held = float(balances.get(coin, 0.0)) * prices.get(coin, 0.0)
        delta = tgt - held
        if abs(delta) < min_notional_safety:
            continue
        price = prices.get(coin)
        if delta > 0:
            if coin in blacklist:
                continue                      # reduce-only: buys always fail
            orders.append({"coin": coin, "side": "buy",
                           "amount_quote": round(delta, 2), "price": price})
        else:
            qty = -delta / max(prices.get(coin, 1.0), 1e-12)
            orders.append({"coin": coin, "side": "sell", "amount": qty,
                           "price": price})
    return orders, value


def place_orders(broker, orders: List[dict]) -> List[dict]:
    """Place orders one by one; a failing order never blocks the rest.

    The reference price fetched by the rebalance is passed through so brokers
    do not re-fetch a ticker per order on the live path."""
    placed = []
    for o in orders:
        try:
            if o["side"] == "buy":
                r = broker.market_buy(o["coin"], o["amount_quote"], price=o.get("price"))
            else:
                r = broker.market_sell(o["coin"], o["amount"], price=o.get("price"))
            placed.append({**o, "result": r})
        except Exception as e:  # noqa: BLE001 - per-order isolation
            placed.append({**o, "error": str(e)})
    return placed


def rebalance(target_w: Dict[str, float], codes: List[str], broker,
              min_notional_safety: float) -> dict:
    """Diff target weights vs holdings and place market orders."""
    from orange_quant import blacklist as blacklist_store

    quote_ccy = broker.quote_ccy
    path = getattr(broker, "blacklist_path", None)
    black = blacklist_store.load(path) if path else set()
    balances = broker.get_balances() or {}
    prices = broker.get_current_prices(codes) or {}
    orders, value = diff_orders(target_w, codes, balances, prices,
                                quote_ccy, min_notional_safety, blacklist=black)
    placed = place_orders(broker, orders)
    print(f"[execute] value={value:.2f} {quote_ccy}, {len(placed)} orders placed")
    return {"orders": placed, "portfolio_value": round(value, 2)}
