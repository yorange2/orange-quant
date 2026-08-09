"""Reinforcement-learning daily rotation — the orange-quant research core.

Market-agnostic: for any frozen universe (A-share top-N liquid names, crypto
top-N coins) the agent picks a holding tier (0..3) per symbol each day; the
resulting weights are executed at next-day open. Raw bars come from local CSVs
(``dataset.py``), the environment is a gym.Env (``env.py``), training uses a
MultiDiscrete PPO built on tianshou (``network.py``/``policy.py``/``train.py``),
and ``backtest.py`` evaluates the trained policy on the test segment.
"""
