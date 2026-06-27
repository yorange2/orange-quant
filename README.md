# 🍊 Orange Quant

基于 [Microsoft qlib](https://github.com/microsoft/qlib) 的 AI 量化交易算法框架。

## 项目结构

```
orange-quant/
├── orange_quant/           # 核心包
│   ├── data/               # 数据层：下载 + Alpha 因子
│   ├── models/             # 模型层：LightGBM 训练器
│   ├── strategies/         # 策略层：基类 + 动量TopK示例
│   ├── backtest/           # 回测层：回测运行器
│   └── workflow/           # 实验管理：完整实验流程
├── config/
│   └── workflow_config.yaml  # YAML 配置驱动实验
├── notebooks/
│   └── 01_quickstart.ipynb   # 快速入门教程
├── scripts/
│   └── download_data.py      # 数据下载脚本
└── .venv/                    # Python 虚拟环境
```

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
cd orange-quant
python3 -m venv .venv
source .venv/bin/activate
pip install -e /path/to/qlib       # 从本地安装 qlib
pip install lightgbm pandas numpy pyyaml
```

### 2. 下载数据

```bash
source .venv/bin/activate
python scripts/download_data.py
```

首次运行会下载中国A股日线数据（约 1-2 GB），请耐心等待。

### 3. 运行实验

```python
from orange_quant.workflow.experiment import run_from_yaml
results = run_from_yaml("config/workflow_config.yaml")
```

或在 notebook 中打开 `notebooks/01_quickstart.ipynb` 逐步体验。

## 开发自定义策略

继承 `BaseQuantStrategy`，实现 `generate_trade_decision` 方法：

```python
from orange_quant.strategies.base import BaseQuantStrategy
from qlib.backtest.decision import Order, TradeDecisionWO

class MyStrategy(BaseQuantStrategy):
    def generate_trade_decision(self, execute_result=None):
        # 你的交易逻辑
        orders = [
            Order(stock_id="SH600000", amount=100, direction=Order.BUY),
        ]
        return TradeDecisionWO(orders, self)
```

## 与 qlib 的关系

- 100% 兼容 qlib 生态（数据格式、模型、回测引擎）
- qlib 的 `qrun` CLI 也可直接运行 `config/workflow_config.yaml`
- 所有 qlib 自带的模型（LGBM、CatBoost、LSTM、Transformer 等）都可直接使用

## 许可

MIT
