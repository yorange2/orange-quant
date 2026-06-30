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
| `workflow_config.yaml` | A 股 CSI300 (820) | 0.027 | Alpha158 + LightGBM |
| `workflow_config_crypto.yaml` | Crypto 187 合约 | 0.043 | Hyperliquid 合约成本 |
| `workflow_config_crypto_spot.yaml` | Crypto 9 U现货 | 0.150 | U 封装代币现货 |
| `workflow_config_binance.yaml` | Binance 20 蓝筹 | ~0.02 | Binance 现货成本 |

## 运行

### 单个实验

```bash
source .venv/bin/activate
python -c "
from orange_quant.workflow.experiment import run_from_yaml
results = run_from_yaml('config/workflow_config_crypto_spot.yaml')
r = results['recorder']
print('IC:', {k:v for k,v in r.list_metrics().items() if 'IC' in k})
print('Excess return:', {k:v for k,v in r.list_metrics().items() if 'annualized' in k})
"
```

### Notebook 方式

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_quickstart.ipynb    # A 股入门
jupyter notebook notebooks/02_crypto_lightgbm.ipynb # Crypto 入门
```

### 深度学习实验（LSTM/GRU/Transformer）

```bash
source .venv/bin/activate
python -c "
from orange_quant.workflow.experiment import run_dl_from_yaml
results = run_dl_from_yaml('config/workflow_config_dl_lstm.yaml')
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

## 导出模型

训练完成后，模型自动保存在 `mlruns/` 中。导出用于实盘交易：

```bash
source .venv/bin/activate
python -c "
import qlib, pickle
qlib.init(provider_uri='data/qlib_data/binance', region='cn')
from qlib.workflow import R
exp = R.list_experiments().get('orange_quant_exp')
for rid in exp.info['recorders'][-5:]:
    r = exp.get_recorder(rid)
    if 'lgb_model.pkl' in r.list_artifacts():
        model = r.load_object('lgb_model.pkl')
        pickle.dump(model, open('models/binance20_lgb.pkl', 'wb'))
        print(f'Model exported from {rid}')
"
```
