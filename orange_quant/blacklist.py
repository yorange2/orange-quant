"""
Persistent blacklist of reduce-only assets.

Some venues reject buys of certain assets (e.g. Binance error -2010,
"This order would increase a reduce-only asset position" — coins scheduled for
delisting or restricted by region). Coins recorded here are excluded from signal
rankings so their budget flows to the next-ranked coin instead of sitting idle.

The store is a JSON array of coin codes at a caller-supplied path, so the same
logic serves any exchange that needs it (only Binance does, today).
"""

import json
from pathlib import Path
from typing import Set, Union

_DEFAULT_PATH = "data/reduce_only_blacklist.json"


def load(path: Union[str, Path] = _DEFAULT_PATH) -> Set[str]:
    """Load the blacklisted coin set (empty if no file yet)."""
    try:
        return set(json.loads(Path(path).read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def add(coin: str, path: Union[str, Path] = _DEFAULT_PATH):
    """Add a coin to the blacklist and persist it."""
    p = Path(path)
    coins = load(p)
    if coin not in coins:
        coins.add(coin)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sorted(coins), indent=2) + "\n")
        print(f"[blacklist] ⛔ {coin} is reduce-only, excluded from future buys")
