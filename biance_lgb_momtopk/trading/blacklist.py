"""Reduce-only blacklist — moved to ``orange_quant.blacklist``.

Re-exported here bound to the Binance store path, so existing
``from biance_lgb_momtopk.trading import blacklist`` calls keep working.
"""

from orange_quant import blacklist as _bl

_PATH = "data/reduce_only_blacklist.json"


def load():
    return _bl.load(_PATH)


def add(coin: str):
    return _bl.add(coin, _PATH)
