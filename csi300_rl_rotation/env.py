"""RotationEnv — gym.Env for daily portfolio rotation over the frozen universe.

Timing (no look-ahead):
  * at close of day t the agent observes feats[t] plus its current tier vector
    and picks target tiers for day t+1;
  * execution happens at t+1's open: the *old* weights earn the overnight gap
    (open[t+1]/close[last traded] − 1) and the *new* weights earn the intraday
    return (close[t+1]/open[t+1] − 1), minus turnover cost on weight changes.

Tiers → weights: base weights {0, 1/3, 2/3, 1}; all-zero → full cash; otherwise
normalized to sum 1 with an iterative per-name cap (max_weight) that prevents a
single stock from dominating. With unchanged weights the reward degenerates
exactly to the close-to-close portfolio return — asserted in the self-check.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces

from csi300_rl_rotation.data import RotationDataset


class RotationEnv(gym.Env):
    """A-share daily rotation environment over a frozen top-N universe."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ds: RotationDataset,
        segment: str = "train",
        horizon: Optional[int] = None,
        tiers: Optional[List[float]] = None,
        max_weight: float = 0.10,
        cost_rate: float = 0.001,
        turnover_penalty: float = 0.0,
        baseline_reward: bool = False,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ds = ds
        self.segment = segment
        s0, s1 = ds.split_idx[segment]
        self._split_lo, self._split_hi = int(s0), int(s1)
        self.horizon = horizon if horizon is not None else self._split_hi - self._split_lo + 1
        self.tiers = np.asarray(tiers if tiers is not None else [0, 1 / 3, 2 / 3, 1.0])
        assert len(self.tiers) == 4, "tiers must map 4 discrete levels"
        self.max_weight = max_weight
        self.cost_rate = cost_rate
        self.turnover_penalty = turnover_penalty
        self.baseline_reward = baseline_reward
        self._rng = np.random.default_rng(seed)
        self._start_idx = start_idx
        if start_idx is not None:
            assert self._split_lo <= start_idx < self._split_hi, "start_idx outside segment"

        n = ds.n_stocks
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n * ds.n_feats + n,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(np.array([4] * n, dtype=np.int64))

        self._t: int = 0          # current day index (decision day)
        self._g: np.ndarray       # current tier vector (50,), int
        self._w: np.ndarray       # current weights (50,), float
        self._step_in_ep: int = 0

    # ------------------------------------------------------------------ core
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._start_idx is not None:
            self._t = self._start_idx
        else:
            hi = self._split_hi - self.horizon
            assert self._split_lo <= hi, "segment shorter than horizon"
            self._t = int(self._rng.integers(self._split_lo, hi + 1))
        self._g = np.zeros(self.ds.n_stocks, dtype=np.int64)
        self._w = np.zeros(self.ds.n_stocks, dtype=np.float64)
        self._step_in_ep = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.int64)
        assert action.shape == (self.ds.n_stocks,), f"bad action shape {action.shape}"
        assert (0 <= action).all() and (action < 4).all(), "tier out of range"

        new_w = self.weights_from_tiers(action)
        t1 = self._t + 1
        assert np.isfinite(new_w).all()
        # old weights earn the overnight gap, new weights the intraday move
        ret_gap = float(np.dot(self._w, self.ds.r_gap[t1]))
        ret_intra = float(np.dot(new_w, self.ds.r_intra[t1]))
        turnover = float(np.abs(new_w - self._w).sum())
        cost = float(self.cost_rate * turnover)
        reward = ret_gap + ret_intra - cost - self.turnover_penalty * turnover
        if self.baseline_reward:
            # differential reward vs equal-weight (daily-rebalanced, costless):
            # teaches relative alpha; equivalent optimal policy as pure reward
            reward -= float((self.ds.r_gap[t1] + self.ds.r_intra[t1]).mean())
        assert np.isfinite(reward), f"non-finite reward: {reward}"

        self._g, self._w = action, new_w
        self._t = t1
        self._step_in_ep += 1
        terminated = self._step_in_ep >= self.horizon
        truncated = False
        info = {
            "date": str(self.ds.dates[self._t]),
            "tiers": action.copy(),
            "weights": new_w.copy(),
            "reward_gap": ret_gap,
            "reward_intra": ret_intra,
            "cost": cost,
            "turnover": turnover,
        }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------- helpers
    def _obs(self) -> np.ndarray:
        obs = np.concatenate([
            self.ds.feats[self._t].reshape(-1),
            self._g.astype(np.float32) / 3.0,
        ]).astype(np.float32)
        assert obs.shape == (self.observation_space.shape[0],)
        return obs

    @staticmethod
    def _normalize_tiers(tiers_act: np.ndarray,
                         tier_weights: np.ndarray,
                         max_weight: float) -> np.ndarray:
        """Tier indices → normalized weights with an iterative per-name cap.

        All-zero tiers → full cash (w = 0). Otherwise w = v/Σv, then repeatedly
        pin any name above ``max_weight`` to it and re-distribute the freed mass
        to the remaining names until nothing exceeds the cap.
        """
        v = tier_weights[np.asarray(tiers_act, dtype=np.int64)].astype(np.float64)
        if v.sum() <= 0.0:
            return np.zeros_like(v)
        w = v / v.sum()
        for _ in range(8):  # converges in a few passes
            over = w > max_weight
            if not over.any():
                break
            w[over] = max_weight
            slack = 1.0 - w.sum()
            free = ~over
            if not free.any() or slack <= 0.0:
                break
            w[free] += slack * (v[free] / v[free].sum())
        return w

    def weights_from_tiers(self, tiers_act: np.ndarray) -> np.ndarray:
        return self._normalize_tiers(tiers_act, self.tiers, self.max_weight)


# ---------------------------------------------------------------------------
# self-check: random policy over the cached dataset
# ---------------------------------------------------------------------------
def main() -> None:
    from csi300_rl_rotation.data import load_config, load_or_build

    cfg = load_config("csi300-rl-rotation")
    ds = load_or_build(cfg)
    env = RotationEnv(ds, segment="train", horizon=120, seed=7)

    print("== env self-check ==")
    n, f = ds.n_stocks, ds.n_feats
    assert env.observation_space.shape == (n * f + n,), "obs space shape"
    assert env.action_space.shape == (n,), "action space shape"
    print(f"  ✓ spaces: obs ({n * f + n},), action MultiDiscrete({n}×4)")

    # 100 random episodes: shapes + finite rewards
    rew_sum, ep_len = 0.0, 0
    for ep in range(100):
        obs, _ = env.reset()
        assert obs.shape == (n * f + n,) and np.isfinite(obs).all(), "obs finite"
        done = False
        while not done:
            act = env.np_random.integers(0, 4, size=n)
            obs, rew, term, trunc, info = env.step(act)
            assert np.isfinite(rew), "reward finite"
            w = info["weights"]
            assert np.isfinite(w).all() and (w >= 0).all(), "weights sane"
            if w.sum() > 0:
                assert w.max() <= env.max_weight + 1e-9, "max_weight cap"
            rew_sum += rew
            ep_len += 1
            done = term or trunc
    print(f"  ✓ 100 episodes, {ep_len} steps, mean reward {rew_sum / ep_len:.6f}")

    # weight normalization: cash vs fully invested
    w_cash = env.weights_from_tiers(np.zeros(n, dtype=np.int64))
    assert w_cash.sum() == 0.0, "all-zero tiers must be cash"
    w_full = env.weights_from_tiers(np.full(n, 3, dtype=np.int64))
    assert abs(w_full.sum() - 1.0) < 1e-9 and w_full.max() <= env.max_weight + 1e-9
    print(f"  ✓ tiers→weights: cash={w_cash.sum():.3f}, full={w_full.sum():.3f} "
          f"(max {w_full.max():.3f} ≤ {env.max_weight})")

    # reward decomposition under an existing holding: zero turnover ⇒
    # reward == w · (r_gap + r_intra) (close-to-close), no cost
    env.reset()
    obs, _, *_ = env.step(env.np_random.integers(0, 4, size=n))  # establish a holding
    held_g = env._g.copy()
    w_held = env.weights_from_tiers(held_g)
    obs, rew, *_ = env.step(held_g)  # no change → no cost, no gap exposure change
    t = env._t
    expect = float(np.dot(w_held, env.ds.r_gap[t] + env.ds.r_intra[t]))
    assert abs(rew - expect) < 1e-6, f"zero-turnover mismatch: {rew} vs {expect}"
    print(f"  ✓ zero-turnover reward matches close-to-close "
          f"({rew:.6f} ≈ {expect:.6f}, held {int(w_held.sum())} names)")

    print("== env self-check PASS ==")


if __name__ == "__main__":
    main()
