"""PPO training entry for the csi300 RL rotation strategy.

Run: cd orange-quant && ../.venv/bin/python -m orange_quant.rl.train <config>

Data is read from the npz cache (build it first with
``python -m orange_quant.rl.dataset <config>``). Each epoch collects
step_per_collect transitions from num_envs parallel training envs (random
episode starts), runs repeat_per_collect PPO epochs, then evaluates the policy
on fixed-seed validation envs; the best validation checkpoint is saved to
models/<cfg>/policy_best.pth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import onpolicy_trainer

from orange_quant.rl.dataset import load_config, load_or_build
from orange_quant.rl.env import RotationEnv
from orange_quant.rl.network import MultiDiscreteActor, RotationCritic
from orange_quant.rl.policy import MultiDiscretePPO


def make_envs(ds, segment: str, horizon: int, env_cfg: dict, n: int,
              seed_base: Optional[int], turnover_penalty: float = 0.0,
              baseline_reward: bool = False):
    """Build a vector env. The turnover penalty and the differential-reward
    baseline guide training only — valid/test evaluation uses pure returns so
    best-model selection and backtest reflect real (cost-net) performance."""
    return DummyVectorEnv([
        (lambda i=i: RotationEnv(
            ds, segment=segment, horizon=horizon,
            tiers=env_cfg["tiers"], max_weight=env_cfg["max_weight"],
            cost_rate=env_cfg["cost_rate"],
            turnover_penalty=turnover_penalty,
            baseline_reward=baseline_reward,
            decision_every=env_cfg.get("decision_every", 1),
            seed=None if seed_base is None else seed_base + i,
        ))
        for i in range(n)
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train csi300 RL rotation PPO")
    parser.add_argument("config", help="config name without .yaml")
    parser.add_argument("--max-epoch", type=int, default=None,
                        help="override ppo.max_epoch (quick smoke runs)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg)
    ppo, model_cfg, env_cfg, paths = cfg["ppo"], cfg["model"], cfg["env"], cfg["paths"]
    if args.max_epoch:
        ppo = {**ppo, "max_epoch": args.max_epoch}
    seed = int(ppo["seed"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(model_cfg["device"])

    n_envs = int(ppo["num_envs"])
    train_envs = make_envs(ds, "train", env_cfg["horizon"], env_cfg, n_envs,
                           seed,
                           turnover_penalty=env_cfg.get("turnover_penalty", 0.0),
                           baseline_reward=env_cfg.get("baseline_reward", False))
    n_test = int(ppo["test_episodes"])
    test_envs = make_envs(ds, "valid", env_cfg["horizon"], env_cfg, n_test,
                          seed, turnover_penalty=0.0, baseline_reward=False)

    obs_dim = ds.n_stocks * ds.n_feats + ds.n_stocks
    hidden = tuple(model_cfg["hidden"])
    actor = MultiDiscreteActor(obs_dim, ds.n_stocks, len(env_cfg["tiers"]), hidden).to(device)
    critic = RotationCritic(obs_dim, hidden).to(device)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=ppo["lr"]
    )
    policy = MultiDiscretePPO(
        actor, critic, optim,
        discount_factor=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        eps_clip=ppo["eps_clip"],
        vf_coef=ppo["vf_coef"],
        ent_coef=ppo["ent_coef"],
        max_grad_norm=ppo["max_grad_norm"],
        max_batchsize=ppo["max_batchsize"],
        reward_normalization=ppo["reward_normalization"],
        value_clip=ppo["value_clip"],
        deterministic_eval=True,
        hold_bias=ppo.get("hold_bias", 0.0),
        action_space=train_envs.action_space[0],  # 0.4.10: per-env list
        observation_space=train_envs.observation_space[0],
    )

    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(int(ppo["step_per_epoch"]), n_envs),
    )
    test_collector = Collector(policy, test_envs, None)

    model_dir = Path(paths["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = model_dir / "policy_best.pth"
    best_metric_path = model_dir / "best_metric.json"
    best_reward = -np.inf
    best_epoch = 0

    def evaluate_valid(epoch: int) -> float:
        nonlocal best_reward, best_epoch
        policy.eval()
        with torch.no_grad():
            result = test_collector.collect(n_episode=n_test)
        policy.train()
        rew = float(np.asarray(result["rew"]).mean())
        print(f"[train] epoch {epoch}: valid mean reward {rew:.6f} "
              f"(± {float(result['rew_std']):.6f}, {int(result['n/ep'])} eps)")
        if rew > best_reward:
            best_reward, best_epoch = rew, epoch
            torch.save(policy.state_dict(), best_ckpt)
            best_metric_path.write_text(json.dumps(
                {"epoch": epoch, "valid_mean_reward": rew}, indent=2))
            print(f"[train]  → best checkpoint saved: {best_ckpt}")
        return rew

    def train_fn(epoch: int, env_step: int) -> None:
        evaluate_valid(epoch)

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=int(ppo["max_epoch"]),
        step_per_epoch=int(ppo["step_per_epoch"]),
        repeat_per_collect=int(ppo["repeat_per_collect"]),
        episode_per_test=n_test,
        batch_size=int(ppo["batch_size"]),
        step_per_collect=int(ppo["step_per_collect"]),
        train_fn=train_fn,
        test_in_train=False,
        show_progress=False,
        verbose=False,
    )

    # final checkpoint (best is preferred; save both)
    torch.save(policy.state_dict(), model_dir / "policy_final.pth")
    print(f"[train] done: {int(result['train_step'])} steps collected, "
          f"best valid reward {best_reward:.6f} (epoch {best_epoch})")

    if not args.no_mlflow:
        try:
            import os
            # mlflow 3.x blocks the filesystem backend by default
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            import mlflow
            with mlflow.start_run(run_name="csi300-rl-rotation"):
                mlflow.log_params({**ppo, "device": model_cfg["device"],
                                   "hidden": list(hidden),
                                   "universe_top_n": ds.n_stocks})
                mlflow.log_metric("valid_best_reward", best_reward)
                mlflow.log_metric("valid_best_epoch", best_epoch)
                mlflow.log_artifact(str(best_ckpt))
                print(f"[train] mlflow run logged")
        except Exception as e:  # noqa: BLE001 - tracking is best-effort
            print(f"[train] mlflow logging skipped: {e}")


if __name__ == "__main__":
    main()
