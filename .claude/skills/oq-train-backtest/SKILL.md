---
name: oq-train-backtest
description: 训练 LightGBM 模型并运行回测，输出 IC、收益率、夏普等指标
---

# Train & Backtest

训练 LightGBM 模型并运行回测，输出 IC、收益率、夏普等指标。

## 触发条件
- "训练模型" / "回测" / "跑实验" / "train" / "backtest"
- "评估策略" / "看看效果"

## 实验配置

| 配置 | 标的 | IC | 说明 |
|------|------|-----|------|
| `csi300-lgb-momtopk.yaml` | A 股 CSI300 (820) | 0.027 | Alpha158 + LightGBM |
| `binance-lgb-momtopk.yaml` | Binance 20 蓝筹 | ~0.02 | Binance 现货成本 |

## 运行

### 单个实验

```bash
source .venv/bin/activate
python -c "
from orange_quant.workflow.experiment import run_from_yaml
results = run_from_yaml('config/binance-lgb-momtopk.yaml')
r = results['recorder']
print('IC:', {k:v for k,v in r.list_metrics().items() if 'IC' in k})
print('Excess return:', {k:v for k,v in r.list_metrics().items() if 'annualized' in k})
"
```

### 深度学习实验（LSTM/GRU/Transformer）

```bash
source .venv/bin/activate
python -c "
from orange_quant.workflow.experiment import run_dl_from_yaml
results = run_dl_from_yaml('config/csi300-lstm-momtopk.yaml')
"
```

> 注意：DL 训练需要 GPU，CPU 训练极慢。

## 输出指标

- **IC** (>0.05 有效因子，>0.1 优秀)
- **ICIR** (>0.5 稳定，>1.0 优秀)
- **Rank IC** (排名相关系数，更稳健)
- **年化超额收益** (含/不含交易成本)
- **Information Ratio** (>1.0 优秀)
- **最大回撤**

## 模型导出

训练完成后，模型**自动导出**到 `models/` 目录，文件名与配置一致：

```bash
# 训练后自动生成:
config/binance-lgb-momtopk.yaml  →  models/binance-lgb-momtopk.pkl
config/csi300-lgb-momtopk.yaml   →  models/csi300-lgb-momtopk.pkl
config/csi300-lstm-momtopk.yaml  →  models/csi300-lstm-momtopk.pkl
```

实盘交易直接加载对应 pkl 文件即可。
