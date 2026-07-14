"""
Persistent blacklist of reduce-only assets.

Binance rejects buys of some assets with error -2010
("This order would increase a reduce-only asset position"),
e.g. coins scheduled for delisting or restricted by region.
Coins recorded here are excluded from signal rankings so their
budget flows to the next-ranked coin instead of sitting idle.
"""

import json
from pathlib import Path
from typing import Set

_PATH = Path(__file__).resolve().parents[2] / "data" / "reduce_only_blacklist.json"


def load() -> Set[str]:
    """Load the blacklisted coin set (empty if no file yet)."""
    try:
        return set(json.loads(_PATH.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def add(coin: str):
    """Add a coin to the blacklist and persist it."""
    coins = load()
    if coin not in coins:
        coins.add(coin)
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(sorted(coins), indent=2) + "\n")
        print(f"[blacklist] ⛔ {coin} is reduce-only, excluded from future buys")
