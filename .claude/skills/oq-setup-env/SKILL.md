---
name: oq-setup-env
description: Configure the orange-quant environment — venv, tianshou/gym pinning, protobuf fix, API keys
---

# Setup Environment

## Python environment

共享 venv 在 **workspace 根目录** `.venv/`（Python 3.12，homebrew）——orange-quant/ 内不自建：

```bash
cd orange-quant
source ../.venv/bin/activate
pip install -e .          # pyproject deps: tianshou/gym==0.26.2/torch/ccxt/akshare/mlflow/...
```

## Known pitfalls

1. **protobuf**: installing `tianshou` downgrades protobuf to 3.19.6, which
   breaks mlflow (needs ≥5.0) → qlib-free code is fine, but fix anyway:
   `pip install protobuf==5.29.6` (skips databricks-sdk's excluded 5.26–5.29.4).
2. **gym is pinned to 0.26.2** — tianshou 0.4.10 requires it; do not upgrade
   to gymnasium (numpy deprecation warnings are noise).
3. **mlflow file store** is blocked by default in 3.x: set
   `MLFLOW_ALLOW_FILE_STORE=true` (train.py already setdefaults it; Docker sets it).
4. **cwd**: always run from the `orange-quant/` directory（workspace 根的 `qlib/` 文件夹会遮蔽已安装的 pyqlib，见 CLAUDE.md 坑 1）。

## API keys (.env, gitignored)

```bash
BINANCE_API_KEY=...
BIANCE_SECRET_KEY=...
HYPERLIQUID_ADDRESS=...
HYPERLIQUID_PRIVATE_KEY=...
```

## Sanity check

```bash
../.venv/bin/python -m orange_quant.rl.smoke_test    # tianshou × gym × torch × numpy compat + PPO flow
```
