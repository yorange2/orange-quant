# 🍊 Orange Quant

基于 [Microsoft qlib](https://github.com/microsoft/qlib) 的 AI 量化交易算法框架。

## 项目结构

```
orange-quant/
├── orange_quant/               # 核心包
│   ├── data/                   # 数据层：下载 + Alpha 因子
│   ├── models/                 # 模型层：LightGBM + DL 训练器
│   ├── strategies/             # 策略层：动量TopK 策略
│   ├── backtest/               # 回测层：回测运行器
│   ├── trading/                # 交易层：Binance 自动交易
│   └── workflow/               # 实验管理：YAML 配置驱动
├── config/                     # 实验配置文件
│   ├── workflow_config.yaml            # A 股 CSI300
│   ├── workflow_config_crypto.yaml     # Crypto 187 合约
│   ├── workflow_config_crypto_spot.yaml # Crypto 9 现货
│   └── workflow_config_binance.yaml    # Binance 20 蓝筹
├── notebooks/                  # Jupyter 教程
├── scripts/                    # 工具脚本
│   ├── build_crypto_data.py        # 构建 Crypto 数据集
│   ├── build_binance_data.py       # 构建 Binance 数据集
│   ├── server_entrypoint.py         # Docker 交易服务入口
│   └── run_binance_testnet.py      # 交易测试
├── Dockerfile                  # Docker 镜像
├── docker-compose.yml          # 一键部署
└── requirements.txt
```

## 快速开始（本地开发）

### 1. 安装依赖

```bash
cd orange-quant
python3 -m venv .venv
source .venv/bin/activate
pip install -e /path/to/qlib              # qlib 本地安装
pip install lightgbm pandas numpy pyyaml ccxt python-dotenv
brew install libomp                       # macOS: LightGBM 依赖
```

### 2. 下载数据并运行实验

```bash
# A 股数据
python scripts/download_data.py
python -c "from orange_quant.workflow.experiment import run_from_yaml; run_from_yaml('config/workflow_config.yaml')"

# Crypto 数据（187 币种，Hyperliquid 合约）
python scripts/build_crypto_data.py
python -c "from orange_quant.workflow.experiment import run_from_yaml; run_from_yaml('config/workflow_config_crypto.yaml')"

# Binance 现货数据（50 币种）
python scripts/build_binance_data.py --top 50
python -c "from orange_quant.workflow.experiment import run_from_yaml; run_from_yaml('config/workflow_config_binance.yaml')"
```

## 实验结果

| 数据集 | 标的 | IC | 超额(含成本) | IR |
|--------|------|-----|-------------|-----|
| A 股 CSI300 | 820 | 0.027 | 1.0% | — |
| Crypto 187 合约 | 187 | 0.043 | 49.5% | 1.75 |
| Crypto 9 U现货 | 9 | 0.150 | 187.3% | 3.74 |
| Binance 20 蓝筹 | 20 | ~0.02 | 17.3% | 0.77 |

## Docker 部署（自动交易）

### 准备

```bash
git clone https://github.com/yorange2/orange-quant.git
cd orange-quant

# 创建 .env 文件（Binance API key）
cat > .env << EOF
BINANCE_API_KEY=你的api_key
BIANCE_SECRET_KEY=你的secret_key
EOF
```

### 启动

```bash
# Live 模式（实盘交易，每日 UTC 00:15 = 北京时间 08:15 调仓）
docker compose up -d

# Dry Run 模式（只分析不下单，先观察）
docker compose --profile dry-run up -d orange-quant-dry

# 查看日志
docker logs -f orange-quant

# 停止
docker compose down
```

### 自定义调仓时间

```bash
# 北京时间 00:30 调仓（UTC 16:30）
docker run -d --env-file .env orange-quant:latest --hour 16 --minute 30

# 只执行一次（测试用）
docker run --rm --env-file .env orange-quant:latest --once --dry-run
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hour` | 0 | 调仓时间 (UTC 小时) |
| `--minute` | 15 | 调仓时间 (分钟) |
| `--topk` | 5 | 持仓数量 |
| `--dry-run` | — | 只分析不下单 |
| `--once` | — | 执行一次后退出 |
| `--testnet` | — | 使用 Binance 测试网 |

## 开发自定义策略

继承 `BaseQuantStrategy`，实现 `generate_trade_decision` 方法：

```python
from orange_quant.strategies.base import BaseQuantStrategy
from qlib.backtest.decision import Order, TradeDecisionWO

class MyStrategy(BaseQuantStrategy):
    def generate_trade_decision(self, execute_result=None):
        orders = [
            Order(stock_id="SH600000", amount=100, direction=Order.BUY),
        ]
        return TradeDecisionWO(orders, self)
```

## 许可

MIT
