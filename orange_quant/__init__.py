"""
Orange Quant shared core.

Exchange-agnostic trading/experiment logic shared by the per-exchange packages
(``biance_lgb_momtopk``, ``hyperliquid_lgb_momtopk``). Each exchange package
provides a thin adapter (broker + :class:`~orange_quant.spec.ExchangeSpec`) and
delegates to this core, so a fix here lands on every exchange at once.
"""

__all__ = ["ExchangeSpec"]

from orange_quant.spec import ExchangeSpec
