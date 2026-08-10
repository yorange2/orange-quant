"""Shared diff-and-execute portfolio rebalancing (RL rotation + LGB rotation).

Extracted from ``orange_quant.live.RLRotationRunner._execute`` so the two
runners place orders identically: diff target weights against current
holdings, skip dust-sized deltas, place market orders with per-order
try/except isolation.
"""

from __future__ import annotations

from typing import Dict, List


def diff_orders(target_w: Dict[str, float], codes: List[str], balances: Dict[str, float],
                prices: Dict[str, float], quote_ccy: str,
                min_notional_safety: float) -> tuple:
    """Build order list from target weights vs holdings. Returns (orders, value)."""
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
        if delta > 0:
            orders.append({"coin": coin, "side": "buy",
                           "amount_quote": round(delta, 2)})
        else:
            qty = -delta / max(prices.get(coin, 1.0), 1e-12)
            orders.append({"coin": coin, "side": "sell", "amount": qty})
    return orders, value


def place_orders(broker, orders: List[dict]) -> List[dict]:
    """Place orders one by one; a failing order never blocks the rest."""
    placed = []
    for o in orders:
        try:
            if o["side"] == "buy":
                r = broker.market_buy(o["coin"], o["amount_quote"])
            else:
                r = broker.market_sell(o["coin"], o["amount"])
            placed.append({**o, "result": r})
        except Exception as e:  # noqa: BLE001 - per-order isolation
            placed.append({**o, "error": str(e)})
    return placed


def rebalance(target_w: Dict[str, float], codes: List[str], broker, quote_ccy: str,
              min_notional_safety: float) -> dict:
    """Diff target weights vs holdings and place market orders."""
    balances = broker.get_balances() or {}
    prices = broker.get_current_prices(codes) or {}
    orders, value = diff_orders(target_w, codes, balances, prices,
                                quote_ccy, min_notional_safety)
    placed = place_orders(broker, orders)
    print(f"[execute] value={value:.2f} {quote_ccy}, {len(placed)} orders placed")
    return {"orders": placed, "portfolio_value": round(value, 2)}
