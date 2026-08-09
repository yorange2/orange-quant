"""PPO policy for the MultiDiscrete tier action space.

tianshou 0.4.10's ``PGPolicy.forward`` already handles MultiDiscrete correctly:
the action space is classified as "discrete" (``MultiDiscrete``/``Discrete``/
``MultiBinary``), so deterministic eval does ``logits.argmax(-1)`` on the actor's
3D logits and stochastic eval calls ``dist.sample()``. The only custom pieces:

* wiring our ``MultiCategorical`` in as ``dist_fn`` (the bundled tianshou
  Actor cannot even emit per-dimension logits);
* ``hold_bias`` — a state-based action prior that penalizes logits of tiers far
  from the current holding. Unlike a reward-side turnover penalty, this acts
  directly on the network output and stays active in deterministic eval, so the
  argmax (deployment) behaviour is stabilised too — a reward penalty cannot do
  that (the sampling distribution tightens, but the argmax still jumps between
  similar observations).
"""

from __future__ import annotations

from typing import Any

import torch
from tianshou.data import Batch
from tianshou.policy import PPOPolicy

from orange_quant.rl.network import MultiCategorical


class MultiDiscretePPO(PPOPolicy):
    """PPO over per-stock independent Categoricals, with a hold prior."""

    def __init__(
        self,
        actor: Any,
        critic: Any,
        optimizer: Any,
        hold_bias: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optimizer, MultiCategorical, **kwargs)
        self.hold_bias = float(hold_bias)

    def forward(self, batch: Batch, state=None, **kwargs) -> Batch:
        """Forward with the hold prior applied to the logits.

        obs layout: [feats (N×F), current tiers/3 (N)] — the tier part sits in
        the last N entries. For each stock, tier k's logit is reduced by
        ``hold_bias * |k − current_tier|``, so staying put is favored and the
        argmax (deterministic eval) becomes stable under feature noise.
        """
        logits, hidden = self.actor(batch.obs, state=state,
                                    info=batch.get("info"))
        if self.hold_bias > 0 and logits.dim() == 3:
            n = logits.shape[-2]
            cur = batch.obs[..., -n:] * 3.0  # current tiers (B, N), 0..3
            k = torch.arange(logits.shape[-1], device=logits.device)
            dist_to_cur = (k[None, None, :] - cur[..., None]).abs()  # (B, N, K)
            logits = logits - self.hold_bias * dist_to_cur
        dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = logits.argmax(-1)
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)
