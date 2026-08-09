"""csi300 RL rotation strategy.

Reinforcement-learning daily rotation over the CSI300's top-50 liquid names:
each day the agent picks a holding tier (0..3) per stock; the resulting weights
are executed at next-day open. Data is precomputed from qlib into numpy arrays
(``data.py``), the environment is a gym.Env (``env.py``), training uses a
MultiDiscrete PPO built on tianshou (``network.py``/``policy.py``/``train.py``),
and ``backtest.py`` evaluates the trained policy on the test segment.
"""
