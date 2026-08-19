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

# Fraction of free quote currency held back when sizing buys, so taker fees and
# slippage on the last order cannot push the account into "insufficient balance".
CASH_BUFFER = 0.005


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


def _place_one(broker, o: dict) -> dict:
    """Place a single order, capturing a raising broker as an ``error`` field."""
    try:
        if o["side"] == "buy":
            r = broker.market_buy(o["coin"], o["amount_quote"], price=o.get("price"))
        else:
            r = broker.market_sell(o["coin"], o["amount"], price=o.get("price"))
        return {**o, "result": r}
    except Exception as e:  # noqa: BLE001 - per-order isolation
        return {**o, "error": str(e)}


def fit_buys_to_cash(buys: List[dict], free_quote: float,
                     min_notional_safety: float,
                     buffer: float = CASH_BUFFER) -> List[dict]:
    """Scale buy notionals down to the quote currency actually on hand.

    Even with sells placed first the funded amount can fall short of the plan
    (fees, slippage between the reference price and the fill), so buys are
    shrunk proportionally — keeping the relative weights intact — and any that
    fall under the dust threshold are dropped rather than sent to fail."""
    wanted = sum(o["amount_quote"] for o in buys)
    spendable = free_quote * (1.0 - buffer)
    if wanted <= spendable:
        return buys

    scale = spendable / wanted if wanted > 0 else 0.0
    print(f"[execute] buys {wanted:.2f} > cash {free_quote:.2f}, scaling ×{scale:.3f}")
    fitted = []
    for o in buys:
        amt = round(o["amount_quote"] * scale, 2)
        if amt < min_notional_safety:
            print(f"[execute] drop buy {o['coin']}: {amt:.2f} below min notional")
            continue
        fitted.append({**o, "amount_quote": amt})
    return fitted


def place_orders(broker, orders: List[dict],
                 min_notional_safety: float = 0.0) -> List[dict]:
    """Place orders one by one; a failing order never blocks the rest.

    Sells go first so the quote currency they free up is settled before the
    buys spend it — placing in universe order instead lets an early buy exhaust
    the balance and makes every later buy fail with "insufficient balance".

    The reference price fetched by the rebalance is passed through so brokers
    do not re-fetch a ticker per order on the live path."""
    sells = [o for o in orders if o["side"] == "sell"]
    buys = [o for o in orders if o["side"] != "sell"]

    placed = [_place_one(broker, o) for o in sells]
    if buys:
        try:
            free = float((broker.get_free_balances() or {}).get(broker.quote_ccy, 0.0))
            buys = fit_buys_to_cash(buys, free, min_notional_safety)
        except Exception as e:  # noqa: BLE001 - fall back to the unscaled plan
            print(f"[execute] cash re-check failed ({e}), placing buys as planned")
        placed += [_place_one(broker, o) for o in buys]
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
    placed = place_orders(broker, orders, min_notional_safety)
    ok = sum(1 for o in placed if o.get("result") is not None)
    print(f"[execute] value={value:.2f} {quote_ccy}, "
          f"{ok}/{len(placed)} orders filled")
    return {"orders": placed, "portfolio_value": round(value, 2)}


def sweep_out_of_universe(broker, codes: List[str], min_notional: float) -> dict:
    """Sell any held coin outside the strategy universe, so the account only
    ever carries universe coins + quote currency.

    Run before the weight rebalance so the returned USDT participates in the
    same day's deployment. Dust (value < min_notional) is skipped — it is best
    cleaned up via the venue's dust-conversion tool. Coins with no quote pair
    fail inside ``market_sell`` and are reported but never block the rest.
    """
    quote_ccy = broker.quote_ccy
    universe = set(codes)
    balances = broker.get_free_balances() or {}
    stray = sorted(set(balances) - universe - {quote_ccy})
    if not stray:
        print("[sweep] no out-of-universe coins")
        return {"sweep_orders": [], "swept_value": 0.0}

    prices = broker.get_current_prices(stray) or {}
    orders, swept = [], 0.0
    for coin in stray:
        qty = float(balances.get(coin, 0.0))
        value = qty * prices.get(coin, 0.0)
        if value < min_notional:
            print(f"[sweep] skip {coin}: {value:.2f} < {min_notional:.0f} {quote_ccy} (dust)")
            continue
        try:
            r = broker.market_sell(coin, qty, price=prices.get(coin))
            orders.append({"coin": coin, "side": "sell", "amount": qty,
                           "price": prices.get(coin), "result": r})
            swept += value
        except Exception as e:  # noqa: BLE001 - per-order isolation
            orders.append({"coin": coin, "side": "sell", "amount": qty,
                           "error": str(e)})
    print(f"[sweep] sold {len(orders)}/{len(stray)} stray coins ≈ {swept:.2f} {quote_ccy}")
    return {"sweep_orders": orders, "swept_value": round(swept, 2)}
