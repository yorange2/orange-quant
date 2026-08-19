"""Shared trading primitives (coin-based broker interface + paper broker)."""

from pathlib import Path

_VENUES = ("binance", "hyperliquid")


def state_path_for(broker, configured: str) -> Path:
    """Idempotency-state path for this broker, ``.paper`` inserted for paper runs.

    Both runners guard on "already acted today" via this file, and the server
    seeds its scheduler from it. Sharing one path across modes means a paper
    dry-run silently marks the day done for the live server — the live trade
    then never fires, with nothing in the logs to say why. Splitting on
    ``broker.is_paper`` keeps the two ledgers independent.

    ``data/live_state/x.json`` → ``data/live_state/x.paper.json``
    """
    p = Path(configured)
    if not getattr(broker, "is_paper", False):
        return p
    return p.with_suffix(f".paper{p.suffix}")


def make_broker(cfg: dict, kind: str):
    """Build a broker from config — the single venue→broker dispatch point.

    ``kind``: ``"paper"`` (public market data for the config's venue, simulated
    fills), ``"live"`` (real broker for the config's venue), or an explicit
    venue name (``"binance"`` / ``"hyperliquid"``). Unknown venues/kinds raise
    instead of silently falling through to the wrong live venue.
    """
    venue = cfg["market"]["venue"]
    if kind == "paper":
        from orange_quant.trading.paper_broker import PaperBroker

        if venue == "binance":
            from orange_quant.trading.binance_broker import _make_public_exchange as mk
        elif venue == "hyperliquid":
            from orange_quant.trading.hyperliquid_broker import _make_exchange as mk
        else:
            raise ValueError(f"no paper-broker data source for venue '{venue}'")
        return PaperBroker([], cfg["market"]["quote_ccy"], mk)

    name = venue if kind == "live" else kind
    if name == "binance":
        from orange_quant.trading.binance_broker import BinanceBroker

        blacklist_path = cfg.get("trading", {}).get("blacklist")
        return BinanceBroker(**({"blacklist_path": blacklist_path} if blacklist_path else {}))
    if name == "hyperliquid":
        from orange_quant.trading.hyperliquid_broker import HyperliquidBroker

        return HyperliquidBroker()
    raise ValueError(f"unknown broker kind/venue '{name}' (expected one of {_VENUES})")
