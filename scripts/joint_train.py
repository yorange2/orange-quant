#!/usr/bin/env python3
"""Multi-market joint training (ROADMAP R4).

Shares ONE policy across two markets (A-share CSI300 top-9 liquid names +
Binance 9-name historical pool): each epoch collects from both markets'
envs alternately and updates on each buffer — the market mix gives the
shared feature representation cross-sectional diversity (A-shares) and
volatility/trend diversity (crypto). Each market keeps its own z-score
(fit on its own train segment) and its own valid/test segments.

Universe N is unified at 9 (A-share pool truncated to its top-9 by
liquidity) so the obs dims match: N×F + N = 99.

Usage: python -m scripts.joint_train [--max-epoch 50] [--step-per-epoch 4096]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from orange_quant.rl.dataset import load_config, load_or_build, RotationDataset
from orange_quant.rl.env import RotationEnv
from orange_quant.rl.network import MultiDiscreteActor, RotationCritic
from orange_quant.rl.policy import MultiDiscretePPO

N_UNIFIED = 9


def truncate_ds(ds: RotationDataset, n: int) -> RotationDataset:
    """Top-n liquidity names only (codes are liquidity-sorted)."""
    return replace(
        ds,
        codes=ds.codes[:n],
        feats=ds.feats[:, :n].copy(),
        r_gap=ds.r_gap[:, :n].copy(),
        r_intra=ds.r_intra[:, :n].copy(),
        zmean=ds.zmean[:n].copy(),
        zstd=ds.zstd[:n].copy(),
    )


def make_envs(ds, segment, horizon, env_cfg, n_envs, seed_base, penalty):
    return DummyVectorEnv([
        (lambda i=i: RotationEnv(
            ds, segment=segment, horizon=horizon,
            tiers=env_cfg["tiers"], max_weight=env_cfg["max_weight"],
            cost_rate=env_cfg["cost_rate"], turnover_penalty=penalty,
            baseline_reward=env_cfg.get("baseline_reward", False),
            decision_every=env_cfg.get("decision_every", 1),
            obs_noise=env_cfg.get("obs_noise", 0.0),
            seed=seed_base + i,
        ))
        for i in range(n_envs)
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epoch", type=int, default=50)
    ap.add_argument("--step-per-epoch", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--test-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    a_cfg = load_config("csi300-rl-rotation")
    b_cfg = load_config("binance-rl-rotation")
    ds_a = truncate_ds(load_or_build(a_cfg), N_UNIFIED)
    ds_b = truncate_ds(load_or_build(b_cfg), N_UNIFIED)
    print(f"[joint] A-share {ds_a.codes} | Binance {ds_b.codes}")

    assert ds_a.n_feats == ds_b.n_feats == 10
    env_a, env_b = a_cfg["env"], b_cfg["env"]
    obs_dim = N_UNIFIED * 10 + N_UNIFIED
    hidden = tuple(a_cfg["model"]["hidden"])
    device = torch.device(a_cfg["model"]["device"])

    actor = MultiDiscreteActor(obs_dim, N_UNIFIED, 4, hidden).to(device)
    critic = RotationCritic(obs_dim, hidden).to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                             lr=b_cfg["ppo"]["lr"])
    policy = MultiDiscretePPO(
        actor, critic, optim,
        discount_factor=b_cfg["ppo"]["gamma"], gae_lambda=b_cfg["ppo"]["gae_lambda"],
        eps_clip=b_cfg["ppo"]["eps_clip"], vf_coef=b_cfg["ppo"]["vf_coef"],
        ent_coef=b_cfg["ppo"]["ent_coef"], max_grad_norm=b_cfg["ppo"]["max_grad_norm"],
        max_batchsize=b_cfg["ppo"]["max_batchsize"],
        reward_normalization=b_cfg["ppo"]["reward_normalization"],
        value_clip=b_cfg["ppo"]["value_clip"], deterministic_eval=True,
        hold_bias=b_cfg["ppo"].get("hold_bias", 0.0),
        action_space=__import__("gym").spaces.MultiDiscrete(np.array([4] * N_UNIFIED)),
        observation_space=__import__("gym").spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
    )

    n_envs = 4
    pen_a = env_a.get("turnover_penalty", 0.0)
    pen_b = env_b.get("turnover_penalty", 0.0)
    train_envs_a = make_envs(ds_a, "train", env_a["horizon"], env_a, n_envs, seed, pen_a)
    train_envs_b = make_envs(ds_b, "train", env_b["horizon"], env_b, n_envs, seed + 100, pen_b)
    val_envs_a = make_envs(ds_a, "valid", env_a["horizon"], env_a, args.test_episodes, seed, 0.0)
    val_envs_b = make_envs(ds_b, "valid", env_b["horizon"], env_b, args.test_episodes, seed + 100, 0.0)

    col_a = Collector(policy, train_envs_a, VectorReplayBuffer(args.step_per_epoch, n_envs))
    col_b = Collector(policy, train_envs_b, VectorReplayBuffer(args.step_per_epoch, n_envs))
    val_a = Collector(policy, val_envs_a, None)
    val_b = Collector(policy, val_envs_b, None)

    out_dir = Path("models/joint-r4")
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -np.inf

    for ep in range(1, args.max_epoch + 1):
        for col, buf, rep in ((col_a, None, args.repeat), (col_b, None, args.repeat)):
            col.collect(n_step=args.step_per_epoch // 2)
            policy.update(sample_size=0, buffer=col.buffer,
                          batch_size=args.batch_size, repeat=rep)
            col.reset_buffer(keep_statistics=True)

        policy.eval()
        with torch.no_grad():
            ra = float(np.asarray(val_a.collect(n_episode=args.test_episodes)["rew"]).mean())
            rb = float(np.asarray(val_b.collect(n_episode=args.test_episodes)["rew"]).mean())
        policy.train()
        avg = (ra + rb) / 2
        if avg > best:
            best = avg
            torch.save(policy.state_dict(), out_dir / "policy_best.pth")
            (out_dir / "best_metric.json").write_text(json.dumps(
                {"epoch": ep, "valid_a": ra, "valid_b": rb}, indent=2))
        if ep % 5 == 0 or ep == 1:
            print(f"[joint] epoch {ep:3d}: valid A {ra:+.4f} | B {rb:+.4f} | avg {avg:+.4f}")

    torch.save(policy.state_dict(), out_dir / "policy_final.pth")
    print(f"[joint] done: best avg valid {best:.4f} → {out_dir}")


if __name__ == "__main__":
    main()
