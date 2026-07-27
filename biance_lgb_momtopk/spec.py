"""Binance ExchangeSpec — wires the adapter into the shared core."""

from orange_quant.spec import ExchangeSpec
from biance_lgb_momtopk.trading.broker import BinanceBroker, PaperBroker, BLACKLIST_PATH
from biance_lgb_momtopk.data import load_coins, rebuild_data

SPEC = ExchangeSpec(
    name="binance",
    quote_ccy="USDT",
    provider_uri="data/qlib_data/binance",
    logger_name="orange-quant",
    server_title="Orange Quant Binance",
    default_model="models/binance-lgb-momtopk.pkl",
    live_broker_factory=BinanceBroker,
    paper_broker_factory=PaperBroker,
    load_coins=load_coins,
    rebuild_data=rebuild_data,
    # Binance does full daily rotation to top-k (ignores the model YAML's
    # n_drop/hold_thresh), applies the reduce-only blacklist, no liquidity filter.
    honor_config_rotation=False,
    liquidity_multiple=0.0,
    use_blacklist=True,
    blacklist_path=BLACKLIST_PATH,
    entry_state_path="data/binance_entry_dates.json",
    default_topk=5,
    default_risk_degree=0.95,
    default_lookback=160,
    default_min_trade=20.0,
    max_position_pct=0.25,
)
