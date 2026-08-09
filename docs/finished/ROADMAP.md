# Orange Quant Roadmap：去 qlib 全面重构

> 创建于 2026-08-09。上一版路线图（qlib 目录重构 + MPS 支持）已完成并被本路线图取代。

## 已完成（2026-08-09 重构落地）

### R1. 删除 qlib 依赖（全部完成）
- 删除 qlib 实验流水线：`experiment.py` / `model_predictor.py` / `ensemble.py` / `spec.py` / `runner.py` / `train.py`（旧 LGB/DL 版）
- 删除适配包：`biance_lgb_momtopk/`、`hyperliquid_lgb_momtopk/`、`csi300_rl_rotation/`（迁入核心）
- 删除研究脚本 `scripts/`（phase_study / strategy_sweep / csi300 build）
- 源码零 qlib import（grep 验证）；pyproject 去 pyqlib/lightgbm
- 旧代码归档：git tag `legacy-pre-rl-refactor`

### R2. 数据层直连（全部完成）
- `data/tencent.py`：A 股日线腾讯直连，**end 锚定分页**（修复旧脚本 2021 年数据空洞 bug）
- `data/universe.py`：流动性冻结 universe（cn/crypto 共用，零额外 API）
- `data/pipeline.py`：去 qlib 二进制构建，保留增量下载
- 删除 qlib bin 写入器（merge_cn_update）与 Yahoo 衔接逻辑

### R3. tianshou 唯一训练范式（全部完成）
- `rl/` 核心：MultiDiscrete PPO（tianshou 0.4.10），A 股 + 加密共用
- 回测/评估：`rl/backtest.py` + `metrics.py`，基准从本地 CSV 读取（SH000300 / BTC）
- 训练/回测/验证全链路零 qlib

### R4. 加密实盘（全部完成）
- `trading/binance_broker.py` / `hyperliquid_broker.py` 迁入核心（ccxt）
- `live.py`：每日 RL 执行 runner（幂等 state_file、min_notional 过滤、黑名单）
- `server.py`：--config 调度循环（--once/--dry-run/心跳/看门狗）
- Docker 容器化（python:3.12-slim，MLFLOW_ALLOW_FILE_STORE）

## 待办

- [ ] A 股研究：腾讯 hfq 新数据训练结果与旧基线（legacy tag 的 outputs）对比复盘
- [ ] 加密实盘小额定单人工确认（先 binance 后 hyperliquid）
- [ ] 加密 universe 幸存偏差：可选引入"历史时点流动性冻结"开关
- [ ] 特征扩展：amount 列 2020 年后可用，可增强流动性/资金流因子
- [ ] 训练稳定性：valid 评估加大 episode 数 / early stopping

## 环境坑备忘（供后续会话）

- protobuf：装 tianshou 后必须 `pip install protobuf==5.29.6`（mlflow 兼容）
- gym 钉死 0.26.2（tianshou 0.4.10 要求）
- tianshou 0.4.10：`DummyVectorEnv.action_space` 是 per-env list（取 `[0]`）
- 腾讯 API 是 end 锚定窗口（640 行/批），分页必须 end 回退
- mlflow 3.x 文件后端需 `MLFLOW_ALLOW_FILE_STORE=true`
