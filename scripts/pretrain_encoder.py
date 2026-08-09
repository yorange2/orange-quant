#!/usr/bin/env python3
"""Supervised encoder pretraining (ROADMAP R3).

Task: predict each symbol's next-bar return from (feats[t], zero tiers) with a
regression head on top of the shared MLP body. The learned body weights are
saved (body.state_dict()) and can be loaded into the RL actor/critic to warm
their feature representation — RL then only has to learn the policy head.

Input layout matches the RL obs (feats flattened + N tier entries set to 0) so
the body's state_dict loads directly into MultiDiscreteActor/RotationCritic.

Usage: python -m scripts.pretrain_encoder [config] [--epochs 50] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from orange_quant.rl.dataset import load_config, load_or_build
from orange_quant.rl.network import _MLP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="binance-rl-rotation")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg)
    model_cfg = cfg["model"]
    hidden = tuple(model_cfg["hidden"])

    tr_s, tr_e = ds.split_idx["train"]
    va_s, va_e = ds.split_idx["valid"]
    n = ds.n_stocks
    nf = ds.n_feats

    def make_xy(s0: int, s1: int):
        # X: (T, N*F + N) with tiers zeroed; y: (T, N) next-bar return
        feats = ds.feats[s0 : s1 + 1]
        T = feats.shape[0]
        X = np.concatenate([feats.reshape(T, -1), np.zeros((T, n), np.float32)], axis=1)
        gap = ds.r_gap[s0 + 1 : s1 + 2]
        intra = ds.r_intra[s0 + 1 : s1 + 2]
        y = (gap + intra).astype(np.float32)
        return X, y

    X_tr, y_tr = make_xy(tr_s, tr_e)
    X_va, y_va = make_xy(va_s, va_e)
    print(f"[pretrain] train {X_tr.shape[0]} bars, valid {X_va.shape[0]} bars")

    body = _MLP(n * nf + n, hidden)
    head = nn.Linear(hidden[-1], n)
    model = nn.Sequential(body, head)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                        batch_size=args.batch)

    best_val = np.inf
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        tr_loss = tot / len(X_tr)

        model.eval()
        with torch.no_grad():
            vloss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in val_dl) / len(X_va)
        if vloss < best_val:
            best_val = vloss
        if ep % 10 == 0 or ep == 1:
            print(f"[pretrain] epoch {ep:3d}: train MSE {tr_loss:.6f} | valid MSE {vloss:.6f}")

    out = Path(args.out if args.out else f"{cfg['paths']['model_dir']}/encoder_body.pth")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(body.state_dict(), out)
    print(f"[pretrain] body saved → {out} (valid MSE {best_val:.6f})")


if __name__ == "__main__":
    main()
