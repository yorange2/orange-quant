# 🍊 Orange Quant

基于 [Microsoft qlib](https://github.com/microsoft/qlib) 的 AI 量化交易算法框架。

## 项目结构

```
orange-quant/
├── orange_quant/               # 核心包
│   ├── trading/                # 交易层：Binance 自动交易
│   └── workflow/               # 实验管理：YAML 配置驱动
├── config/                     # 实验配置文件
│   ├── csi300-lgb-momtopk.yaml
│   └── binance-lgb-momtopk.yaml
├── scripts/                    # 工具脚本
│   ├── biance/
│   │   ├── build_data.py       # 构建 Binance 数据集
│   │   └── execute.py          # 交易执行入口
│   └── csi300/
│       └── build_data.py       # 下载 A 股数据
├── Dockerfile
└── docker-compose.yml
```

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/microsoft/qlib.git
pip install lightgbm pandas numpy pyyaml ccxt python-dotenv
# macOS: brew install libomp
```

### 2. 下载数据并运行实验

```bash
# A 股
python scripts/csi300/build_data.py
python -c "from orange_quant.workflow.experiment import run_from_yaml; run_from_yaml('config/csi300-lgb-momtopk.yaml')"

# Binance
python scripts/biance/build_data.py --top 50
python -c "from orange_quant.workflow.experiment import run_from_yaml; run_from_yaml('config/binance-lgb-momtopk.yaml')"
```

## 实验结果

| 数据集 | 标的 | IC | 超额(含成本) | IR(含成本) |
|--------|------|-----|-------------|-----------|
| A 股 CSI300 | 820 | 0.027 | 1.0% | 0.11 |
| Binance 20 蓝筹 | 20 | 0.034 | 17.3% | 0.77 |

## 自动交易

### 本地

```bash
source .venv/bin/activate

# DRY RUN（只分析不下单）
python scripts/biance/execute.py --once --dry-run --model models/binance-lgb-momtopk.pkl

# 实盘下单
python scripts/biance/execute.py --once --model models/binance-lgb-momtopk.pkl

# 查看持仓
python -c "
from dotenv import load_dotenv; load_dotenv(override=True)
from orange_quant.trading.broker import BinanceBroker
broker = BinanceBroker(testnet=False, paper=False)
for a, amt in sorted(broker.get_balances().items()):
    if amt > 0.0001:
        print(f'  {a}: {amt:.6f}')
"
```

### Docker 部署

```bash
# 准备 .env（Binance API Key）
cat > .env << EOF
BINANCE_API_KEY=你的api_key
BIANCE_SECRET_KEY=你的secret_key
EOF

# 放入模型文件
mkdir -p models
cp /path/to/binance-lgb-momtopk.pkl models/

# 启动
docker compose up -d                    # 实盘
docker compose --profile dry-run up -d  # 观察
docker logs -f orange-quant             # 日志
docker compose down                     # 停止
```

### 升级

```bash
git pull
docker compose up -d --build    # 重新构建并替换旧容器
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hour` | 0 | 调仓时间 (UTC 小时) |
| `--minute` | 15 | 调仓时间 (分钟) |
| `--topk` | 5 | 持仓数量 |
| `--lookback` | 160 | 回看天数 |
| `--model` | models/binance-lgb-momtopk.pkl | LightGBM 模型路径，不指定则用动量策略 |
| `--dry-run` | — | 只分析不下单 |
| `--once` | — | 执行一次后退出 |
| `--testnet` | — | 使用 Binance 测试网 |

## 许可

MIT
