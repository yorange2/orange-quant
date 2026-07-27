"""
ExchangeSpec — the single knob that makes the shared core exchange-specific.

The core (`runner`, `server`, `model_predictor`, `data.pipeline`) is written once
and driven entirely by an ``ExchangeSpec`` instance built in each exchange adapter
package. Trade behaviour is preserved bit-for-bit per exchange by configuring the
spec — there is no exchange-specific branch in the core beyond reading these fields.

Behaviour flags encode the two live strategies as they exist today:

- Binance: full daily rotation (``honor_config_rotation=False`` forces
  ``n_drop=∞, hold_thresh=0``), reduce-only blacklist on, no liquidity filter.
- Hyperliquid: partial rotation honoured from the model's YAML
  (``honor_config_rotation=True``), liquidity filter on, no blacklist.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class ExchangeSpec:
    # --- identity ---
    name: str                         # "binance" | "hyperliquid"
    quote_ccy: str                    # settlement currency, e.g. "USDT" | "USDC"
    provider_uri: str                 # qlib data dir, e.g. "data/qlib_data/binance"
    logger_name: str                  # server logger, e.g. "orange-quant"
    server_title: str                 # startup banner line
    default_model: str                # default --model path

    # --- factories / hooks (provided by the adapter package) ---
    live_broker_factory: Callable[[], object]          # () -> live broker
    paper_broker_factory: Callable[[List[str]], object]  # (coins) -> PaperBroker
    load_coins: Callable[[], List[str]]                # () -> traded coin list
    rebuild_data: Callable[[], None]                   # () -> refresh + rebuild qlib data

    # --- runner behaviour (exchange constants, not read from the model YAML) ---
    # When False the server ignores the model YAML's n_drop/hold_thresh and forces
    # full rotation to top-k (binance). When True it honours them (hyperliquid).
    honor_config_rotation: bool = False
    liquidity_multiple: float = 0.0    # 0 disables the liquidity filter
    use_blacklist: bool = False        # reduce-only blacklist (binance only)
    blacklist_path: Optional[str] = None
    entry_state_path: str = "data/entry_dates.json"

    # --- server / runner defaults (overridden by the model YAML, then CLI) ---
    default_topk: int = 5
    default_risk_degree: float = 0.95
    default_n_drop: int = 1
    default_hold_thresh: int = 1
    default_lookback: int = 160
    default_min_trade: float = 20.0
    max_position_pct: float = 0.25
