"""Actor-critic networks and the MultiCategorical distribution for the
csi300 RL rotation strategy.

tianshou 0.4.10's bundled ``Actor`` flattens a MultiDiscrete action space into
a single softmax over the product space, so per-stock tier decisions would be
sampled jointly instead of independently. Instead we emit 3D logits
(B, n_stocks, n_tiers) from a plain MLP head and pair them with
``MultiCategorical`` — n_stocks independent Categoricals whose log_prob and
entropy sum over the stock dimension, which is exactly what PPO's ratio and
entropy terms need.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class MultiCategorical:
    """Joint distribution of ``n_stocks`` independent Categoricals.

    PPO consumes ``log_prob``/``entropy`` as per-transition scalars; the summed
    form is the only correct answer here — a bare Categorical would return
    (B, n_stocks) and silently broadcast against (B,) advantages.
    """

    def __init__(self, logits: torch.Tensor) -> None:  # (B, n_stocks, n_tiers)
        self._dist = torch.distributions.Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        return self._dist.sample()  # (B, n_stocks) int64

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(act).sum(dim=-1)  # (B,)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1)  # (B,)


def _to_torch(obs, device: torch.device) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...]) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiDiscreteActor(nn.Module):
    """MLP actor emitting (B, n_stocks, n_tiers) logits."""

    def __init__(
        self,
        obs_dim: int,
        n_stocks: int,
        n_tiers: int,
        hidden: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.n_stocks = n_stocks
        self.n_tiers = n_tiers
        self.body = _MLP(obs_dim, hidden)
        self.head = nn.Linear(hidden[-1], n_stocks * n_tiers)

    def forward(self, obs, state=None, info=None):
        x = _to_torch(obs, next(self.parameters()).device)
        logits = self.head(self.body(x))
        return logits.view(-1, self.n_stocks, self.n_tiers), state


class RotationCritic(nn.Module):
    """MLP value head → (B, 1)."""

    def __init__(self, obs_dim: int, hidden: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.body = _MLP(obs_dim, hidden)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, obs, state=None, info=None):
        x = _to_torch(obs, next(self.parameters()).device)
        # tianshou's A2C expects the critic to return the plain value tensor
        return self.head(self.body(x))
