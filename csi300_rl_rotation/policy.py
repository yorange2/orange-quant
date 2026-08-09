"""PPO policy for the MultiDiscrete tier action space.

tianshou 0.4.10's ``PGPolicy.forward`` already handles MultiDiscrete correctly:
the action space is classified as "discrete" (``MultiDiscrete``/``Discrete``/
``MultiBinary``), so deterministic eval does ``logits.argmax(-1)`` on the actor's
3D logits and stochastic eval calls ``dist.sample()``. The only custom piece is
wiring our ``MultiCategorical`` in as ``dist_fn`` (the bundled tianshou Actor
cannot even emit per-dimension logits, so the actor must come from
``network.MultiDiscreteActor``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tianshou.policy import PPOPolicy

from csi300_rl_rotation.network import MultiCategorical


class MultiDiscretePPO(PPOPolicy):
    """PPO over per-stock independent Categoricals."""

    def __init__(
        self,
        actor: Any,
        critic: Any,
        optimizer: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optimizer, MultiCategorical, **kwargs)
