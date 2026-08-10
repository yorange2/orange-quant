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
from orange_quant.rl.metrics import bars_per_year, return_metrics
from orange_quant.rl.network import MultiDiscreteActor, RotationCritic
from orange_quant.rl.policy import MultiDiscretePPO


def load_policy(cfg: dict, ds, device, ckpt: str | None = None):
    """Rebuild the policy graph and load a checkpoint (default: best)."""
    model_cfg, env_cfg, paths = cfg["model"], cfg["env"], cfg["paths"]
    obs_dim = ds.n_stocks * ds.n_feats + ds.n_stocks
    hidden = tuple(model_cfg["hidden"])
    actor = MultiDiscreteActor(obs_dim, ds.n_stocks, len(env_cfg["tiers"]), hidden).to(device)
    critic = RotationCritic(obs_dim, hidden).to(device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    policy = MultiDiscretePPO(
        actor, critic, optim,
        discount_factor=0.99, deterministic_eval=True,
        hold_bias=cfg.get("ppo", {}).get("hold_bias", 0.0),
        action_space=gym.spaces.MultiDiscrete(
            np.array([4] * ds.n_stocks, dtype=np.int64)),
        observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(ds.n_stocks * ds.n_feats + ds.n_stocks,), dtype=np.float32),
    )
    model_dir = Path(paths["model_dir"])
    ckpt_path = Path(ckpt) if ckpt else model_dir / "policy_best.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "policy_final.pth"
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    print(f"[backtest] loaded checkpoint: {ckpt_path}")
    return policy.eval()


def benchmark_closes(cfg: dict, dates) -> pd.Series:
    """Benchmark symbol's close series from the local raw CSV, on ``dates``."""
    sym = cfg["market"]["benchmark_symbol"]
    raw = Path(cfg["data"]["raw_dir"]) / f"{sym}.csv"
    idx = pd.read_csv(raw, parse_dates=["date"]).set_index("date").sort_index()
    return idx["close"].reindex(pd.DatetimeIndex(dates))


def load_benchmark(ds, cfg) -> np.ndarray:
    """Benchmark index NAV from the local raw CSV, aligned to the test calendar."""
    test_s, test_e = ds.split_idx["test"]
    c = benchmark_closes(cfg, ds.dates[test_s : test_e + 1])
    rets = c.pct_change(fill_method=None).fillna(0.0).to_numpy()
    return np.cumprod(1.0 + rets)


def rollout(policy, env) -> pd.DataFrame:
    """Deterministic rollout of ``policy`` over ``env`` → one row per step.

    Shared by the single-shot backtest and the walk-forward windows so the NAV
    row schema and the |Δw|/2 turnover definition stay identical.
    """
    policy.eval()
    rows = []
    nav = 1.0
    w_prev = None
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            act = policy(Batch(obs=obs[None])).act[0]
        obs, rew, term, trunc, info = env.step(act)
        w = info["weights"]
        nav *= 1.0 + rew
        rows.append({
            "date": info["date"], "nav": nav, "ret": rew,
            "turnover": float(np.abs(w - (w_prev if w_prev is not None else 0)).sum()) / 2,
            "n_held": int((w > 0).sum()),
            "max_w": float(w.max()) if w.sum() > 0 else 0.0,
            "in_cash": float(w.sum() == 0.0),
        })
        w_prev = w
        done = term or trunc
    return pd.DataFrame(rows)


def plot_navs(navs: dict, dates, title: str, out_path) -> None:
    """3-line normalized NAV chart (shared by the RL and LGB backtests)."""
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, nv in navs.items():
        ax.plot(dates[: len(nv)], nv / nv[0], label=name, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("NAV (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def write_metrics(metrics: dict, out_dir: Path, header: str,
                  signed_prefixes: tuple = ("annual", "excess", "information")) -> None:
    """Persist metrics.json and print the formatted table."""
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n========== {header} ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:+.4f}" if k.startswith(signed_prefixes)
                  else f"  {k:<28} {v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest csi300 RL rotation")
    parser.add_argument("config", help="config name without .yaml")
    parser.add_argument("--ckpt", default=None, help="override checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg)
    device = torch.device(cfg["model"]["device"])
    policy = load_policy(cfg, ds, device, ckpt=args.ckpt)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    test_s, test_e = ds.split_idx["test"]
    dates = ds.dates[test_s : test_e + 1]
    # last day's return needs t+1 data, which ends at the data boundary — the
    # final settlement day is test_e. horizon is in *policy* steps: raw bars
    # divided by decision_every (frame-skip).
    env_cfg = cfg["env"]
    decision_every = int(env_cfg.get("decision_every", 1))
    horizon = (test_e - test_s) // decision_every

    # ---- RL policy rollout (track weights per day for turnover) ----
    env = RotationEnv(ds, segment="test", horizon=horizon,
                      tiers=env_cfg["tiers"], max_weight=env_cfg["max_weight"],
                      cost_rate=env_cfg["cost_rate"],
                      turnover_penalty=0.0,  # no training-style penalty at test
                      decision_every=env_cfg.get("decision_every", 1),
                      start_idx=test_s, seed=0)
    rl = rollout(policy, env)
    rl.to_csv(out_dir / "nav.csv", index=False)
    print(f"[backtest] RL policy: {len(rl)} days, final NAV {rl['nav'].iloc[-1]:.4f}")

    # ---- benchmarks ----
    s = test_s
    # equal-weight on the *raw* bars, resampled to policy steps so the
    # benchmark NAV lines up row-for-row with the RL rollout
    raw_n = horizon * decision_every
    ew_ret = (ds.r_gap[s : s + raw_n] + ds.r_intra[s : s + raw_n]).mean(axis=1)
    ew_nav = np.cumprod(1.0 + ew_ret)[::decision_every]
    bench_nav = load_benchmark(ds, cfg)[::decision_every]

    t = min(len(rl), len(ew_nav), len(bench_nav))
    navs = {
        "RL 策略": rl["nav"].to_numpy()[:t],
        "等权 top50": ew_nav[:t],
        "SH000300": bench_nav[:t],
    }
    bpy = bars_per_year(cfg)
    metrics = return_metrics(
        rl["nav"].to_numpy()[:t],
        benchmark_nav=ew_nav[:t],
        turnover=rl["turnover"].to_numpy()[:t],
        bars_per_year=bpy,
        annualize=bpy,
    )
    metrics["benchmark_equal_weight_nav"] = float(ew_nav[-1])
    metrics["benchmark_csi300_nav"] = float(bench_nav[-1])
    metrics["policy_nav"] = float(rl["nav"].iloc[-1])
    write_metrics(metrics, out_dir, "回测指标 (vs 等权 top50)")

    plot_navs(navs, dates,
              f"csi300 RL rotation backtest ({dates[0].astype('datetime64[D]')} ~ "
              f"{dates[len(rl) - 1].astype('datetime64[D]')})",
              out_dir / "nav.png")
    print(f"\n[backtest] outputs → {out_dir}/")
    print("[backtest] done")


if __name__ == "__main__":
    main()
