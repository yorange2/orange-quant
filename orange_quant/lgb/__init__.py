"""LightGBM momentum top-k strategy (rebuilt on the new architecture).

Market-agnostic core: Alpha158 features (``features``), dataset/backtest/
train/live (mirroring ``orange_quant.rl``), driven by a config yaml with
``strategy.type: lgb`` (``server.py`` dispatches on it).
"""
