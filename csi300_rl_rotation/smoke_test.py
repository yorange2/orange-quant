"""Smoke test: tianshou 0.4.10 × gym 0.26 × torch 2.5 × numpy 2.x compatibility
plus the MultiDiscrete PPO data flow, on a toy environment.

Run: cd orange-quant && ../.venv/bin/python -m csi300_rl_rotation.smoke_test
"""

from __future__ import annotations

import numpy as np
import torch
from gym import spaces
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from csi300_rl_rotation.network import MultiCategorical, MultiDiscreteActor, RotationCritic
from csi300_rl_rotation.policy import MultiDiscretePPO


class ToyEnv:
    """Minimal gym-0.26 env: Box obs, MultiDiscrete action, 2-step episodes."""

    def __init__(self, n_stocks: int = 3, n_tiers: int = 4) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.array([n_tiers] * n_stocks))
        self._t = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._t = 0
        return np.random.uniform(-1, 1, 6).astype(np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.random.uniform(-1, 1, 6).astype(np.float32)
        rew = float(np.random.uniform(-0.05, 0.05))
        done = self._t >= 2
        return obs, rew, done, False, {}


def _assert(cond, msg: str) -> None:
    assert cond, f"SMOKE FAIL: {msg}"
    print(f"  ✓ {msg}")


def main() -> None:
    print("== smoke: imports ==")
    import gym
    import tianshou

    _assert(gym.__version__ == "0.26.2", f"gym {gym.__version__}")
    _assert(tianshou.__version__ == "0.4.10", f"tianshou {tianshou.__version__}")
    _assert(torch.__version__ == "2.5.1", f"torch {torch.__version__}")
    _assert(np.__version__.startswith("2."), f"numpy {np.__version__}")

    print("== smoke: MultiCategorical ==")
    logits = torch.randn(2, 3, 4)
    dist = MultiCategorical(logits)
    s = dist.sample()
    _assert(s.shape == (2, 3) and s.dtype == torch.int64, f"sample shape {tuple(s.shape)}")
    lp = dist.log_prob(s)
    _assert(lp.shape == (2,) and torch.isfinite(lp).all(), f"log_prob shape {tuple(lp.shape)}")
    ent = dist.entropy()
    _assert(ent.shape == (2,), f"entropy shape {tuple(ent.shape)}")

    print("== smoke: PPO collect + update ==")
    n_stocks, n_tiers, obs_dim = 3, 4, 6
    envs = DummyVectorEnv([lambda: ToyEnv() for _ in range(2)])
    actor = MultiDiscreteActor(obs_dim, n_stocks, n_tiers)
    critic = RotationCritic(obs_dim)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-3
    )
    policy = MultiDiscretePPO(
        actor,
        critic,
        optim,
        discount_factor=0.99,
        reward_normalization=True,
        value_clip=True,
        deterministic_eval=True,
        # DummyVectorEnv.action_space is a per-env list in 0.4.10
        action_space=envs.action_space[0],
    )
    buffer = VectorReplayBuffer(200, len(envs))
    collector = Collector(policy, envs, buffer)

    result = collector.collect(n_step=200)
    _assert(result["n/ep"] > 0, f"collected {result['n/ep']} episodes")
    batch, _ = buffer.sample(batch_size=200)
    _assert(batch.act.shape == (200, n_stocks), f"act shape {tuple(batch.act.shape)}")
    _assert(np.issubdtype(batch.act.dtype, np.integer), f"act dtype {batch.act.dtype}")
    _assert(np.isfinite(batch.obs).all(), "obs finite")
    _assert(np.isfinite(batch.rew).all(), "rew finite")

    # logp computed the way PPO.process_fn does it (act converted to torch first)
    with torch.no_grad():
        lp = policy(batch).dist.log_prob(torch.from_numpy(batch.act))
    _assert(lp.shape == (200,), f"collected logp shape {tuple(lp.shape)}")

    # deterministic eval branch (argmax over tier dim)
    policy.eval()
    with torch.no_grad():
        out = policy(batch)
    _assert(out.act.shape == (200, n_stocks), f"det act shape {tuple(out.act.shape)}")
    policy.train()

    stats = policy.update(sample_size=64, buffer=buffer, batch_size=64, repeat=2)
    loss = np.mean(stats["loss"])
    _assert(np.isfinite(loss), f"update loss {loss}")
    print(f"  ✓ update loss={loss:.4f} (clip={np.mean(stats['loss/clip']):.4f} "
          f"vf={np.mean(stats['loss/vf']):.4f} ent={np.mean(stats['loss/ent']):.4f})")

    print("== smoke: PASS ==")


if __name__ == "__main__":
    main()
