# 🍊 Orange Quant

基于 [Microsoft qlib](https://github.com/microsoft/qlib) 的 AI 量化交易算法框架。

## 项目结构

```
orange-quant/
├── orange_quant/               # 核心包



│   ├── trading/                # 交易层：Binance 自动交易
│   └── workflow/               # 实验管理：YAML 配置驱动
├── config/                     # 实验配置文件
│   ├── csi300-lgb-momtopk.yaml            # A 股 CSI300 × LightGBM
│   └── binance-lgb-momtopk.yaml        # Binance 20 蓝筹 × LightGBM
├── scripts/                    # 工具脚本
│   ├── biance/
│   │   ├── build_data.py          # 构建 Binance 数据集
│   │   └── execute.py             # 交易执行入口
│   └── csi300/
│       └── build_data.py          # 下载 A 股数据
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
python scripts/csi300/build_data.py
python -c "from orange_quant.workflow.experiment import run_from_yaml; run_from_yaml('config/csi300-lgb-momtopk.yaml')"

# Binance 现货数据（50 币种）
python scripts/biance/build_data.py --top 50
python -c "from orange_quant.workflow.experiment import run_from_yaml; run_from_yaml('config/binance-lgb-momtopk.yaml')"
```

## 实验结果

| 数据集 | 标的 | IC | 超额(含成本) | IR |
|--------|------|-----|-------------|-----|
| A 股 CSI300 | 820 | 0.027 | 1.0% | — |
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

## 许可

MIT
