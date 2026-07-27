"""Hyperliquid ExchangeSpec — wires the adapter into the shared core."""

from orange_quant.spec import ExchangeSpec
from hyperliquid_lgb_momtopk.trading.broker import HyperliquidBroker, PaperBroker
from hyperliquid_lgb_momtopk.data import load_coins, rebuild_data

SPEC = ExchangeSpec(
    name="hyperliquid",
    quote_ccy="USDC",
    provider_uri="data/qlib_data/hyperliquid",
    logger_name="orange-quant-hl",
    server_title="Orange Quant Hyperliquid",
    default_model="models/hyperliquid-lgb-momtopk.pkl",
    live_broker_factory=HyperliquidBroker,
    paper_broker_factory=PaperBroker,
    load_coins=load_coins,
    rebuild_data=rebuild_data,
    # Hyperliquid honours the model YAML's n_drop/hold_thresh (partial rotation),
    # filters thin books, and has no reduce-only blacklist.
    honor_config_rotation=True,
    liquidity_multiple=50.0,
    use_blacklist=False,
    entry_state_path="data/hl_entry_dates.json",
    default_topk=5,
    default_risk_degree=0.95,
    default_n_drop=1,
    default_hold_thresh=1,
    default_lookback=160,
    default_min_trade=20.0,
    max_position_pct=0.25,
)
