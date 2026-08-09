"""Backtest: deterministic rollout of a trained policy over the test segment.

Run: cd orange-quant && ../.venv/bin/python -m orange_quant.rl.backtest <config>

Loads models/<cfg>/policy_best.pth (falls back to policy_final.pth), rolls out
the whole test segment as a single episode (deterministic argmax actions),
computes the NAV/positions/turnover, benchmarks against equal-weight top50 and
SH000300, and writes outputs/<cfg>/{nav.csv, positions.csv, metrics.json,
nav.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch

from orange_quant.rl.dataset import load_config, load_or_build
from orange_quant.rl.env import RotationEnv
from orange_quant.rl.metrics import return_metrics
from orange_quant.rl.network import MultiDiscreteActor, RotationCritic
from orange_quant.rl.policy import MultiDiscretePPO


def load_policy(cfg: dict, ds, device):
    """Rebuild the policy graph and load the best checkpoint."""
    model_cfg, env_cfg, paths = cfg["model"], cfg["env"], cfg["paths"]
    obs_dim = ds.n_stocks * ds.n_feats + ds.n_stocks
    hidden = tuple(model_cfg["hidden"])
    actor = MultiDiscreteActor(obs_dim, ds.n_stocks, len(env_cfg["tiers"]), hidden).to(device)
    critic = RotationCritic(obs_dim, hidden).to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    policy = MultiDiscretePPO(
        actor, critic, optim,
        discount_factor=0.99, deterministic_eval=True,
        action_space=gym.spaces.MultiDiscrete(
            np.array([4] * ds.n_stocks, dtype=np.int64)),
        observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(ds.n_stocks * ds.n_feats + ds.n_stocks,), dtype=np.float32),
    )
    model_dir = Path(paths["model_dir"])
    ckpt = model_dir / "policy_best.pth"
    if not ckpt.exists():
        ckpt = model_dir / "policy_final.pth"
    state = torch.load(ckpt, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    print(f"[backtest] loaded checkpoint: {ckpt}")
    return policy.eval()


def load_benchmark(ds, cfg) -> np.ndarray:
    """Benchmark index NAV from the local raw CSV, aligned to the test calendar."""
    sym = cfg["market"]["benchmark_symbol"]
    raw = Path(cfg["data"]["raw_dir"]) / f"{sym}.csv"
    idx = pd.read_csv(raw, parse_dates=["date"]).set_index("date").sort_index()
    test_s, test_e = ds.split_idx["test"]
    dates = pd.DatetimeIndex(ds.dates[test_s : test_e + 1])
    c = idx["close"].reindex(dates)
    rets = c.pct_change(fill_method=None).fillna(0.0).to_numpy()
    nav = np.cumprod(1.0 + rets)
    return nav


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest csi300 RL rotation")
    parser.add_argument("config", help="config name without .yaml")
    parser.add_argument("--ckpt", default=None, help="override checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg)
    device = torch.device(cfg["model"]["device"])
    policy = load_policy(cfg, ds, device)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    test_s, test_e = ds.split_idx["test"]
    dates = ds.dates[test_s : test_e + 1]
    # last day's return needs t+1 data, which ends at the data boundary — the
    # final settlement day is test_e (rollout runs test_e - test_s steps)
    horizon = test_e - test_s
    env_cfg = cfg["env"]

    # ---- RL policy rollout (track weights per day for turnover) ----
    env = RotationEnv(ds, segment="test", horizon=horizon,
                      tiers=env_cfg["tiers"], max_weight=env_cfg["max_weight"],
                      cost_rate=env_cfg["cost_rate"],
                      turnover_penalty=0.0,  # no training-style penalty at test
                      start_idx=test_s, seed=0)
    w_prev = None
    rows = []
    nav = 1.0
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            act = policy(Batch(obs=obs[None])).act[0]
        obs, rew, term, trunc, info = env.step(act)
        w = info["weights"]
        turnover = float(np.abs(w - (w_prev if w_prev is not None else 0)).sum()) / 2
        nav *= 1.0 + rew
        rows.append({
            "date": info["date"], "nav": nav, "ret": rew,
            "turnover": turnover, "n_held": int((w > 0).sum()),
            "max_w": float(w.max()) if w.sum() > 0 else 0.0,
            "in_cash": float(w.sum() == 0.0),
        })
        w_prev = w
        done = term or trunc
    rl = pd.DataFrame(rows)
    rl.to_csv(out_dir / "nav.csv", index=False)
    print(f"[backtest] RL policy: {len(rl)} days, final NAV {rl['nav'].iloc[-1]:.4f}")

    # ---- benchmarks ----
    s = test_s
    ew_ret = (ds.r_gap[s : s + horizon] + ds.r_intra[s : s + horizon]).mean(axis=1)
    ew_nav = np.cumprod(1.0 + ew_ret)
    bench_nav = load_benchmark(ds, cfg)

    t = min(len(rl), len(ew_nav), len(bench_nav))
    navs = {
        "RL 策略": rl["nav"].to_numpy()[:t],
        "等权 top50": ew_nav[:t],
        "SH000300": bench_nav[:t],
    }
    metrics = return_metrics(
        rl["nav"].to_numpy()[:t],
        benchmark_nav=ew_nav[:t],
        turnover=rl["turnover"].to_numpy()[:t],
    )
    metrics["benchmark_equal_weight_nav"] = float(ew_nav[-1])
    metrics["benchmark_csi300_nav"] = float(bench_nav[-1])
    metrics["policy_nav"] = float(rl["nav"].iloc[-1])
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n========== 回测指标 (vs 等权 top50) ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:+.4f}" if k.startswith(("annual", "excess", "information"))
                  else f"  {k:<28} {v:.4f}")

    # ---- plot ----
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, nv in navs.items():
        ax.plot(dates[: len(nv)], nv / nv[0], label=name, linewidth=1.5)
    ax.set_title(f"csi300 RL rotation backtest ({dates[0].astype('datetime64[D]')} ~ "
                 f"{dates[len(rl) - 1].astype('datetime64[D]')})")
    ax.set_ylabel("NAV (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "nav.png", dpi=150)
    print(f"\n[backtest] outputs → {out_dir}/")
    print("[backtest] done")


if __name__ == "__main__":
    main()
