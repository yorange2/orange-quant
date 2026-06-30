# Live Trading

实盘/模拟自动交易。支持 Docker 长期运行和本地测试。

## 触发条件
- "实盘交易" / "下单" / "调仓" / "trade"
- "启动交易服务器" / "start trading"
- "查看持仓" / "check positions"

## 前置条件

1. `.env` 文件已配置 `BINANCE_API_KEY` 和 `BIANCE_SECRET_KEY`
2. Docker 已安装并运行

## Docker 部署（推荐，长期运行）

### 构建并启动

```bash
cd /Users/yuanchengcheng/Documents/GitHub/orange-quant
docker compose up -d --build
```

### 常用操作

```bash
# 查看日志
docker logs -f orange-quant

# 手动执行一次调仓
docker run --rm --env-file .env orange-quant:latest --once

# DRY RUN 模式（只分析不下单）
docker run --rm --env-file .env orange-quant:latest --once --dry-run

# 使用 LightGBM 模型预测（动量默认）
docker run --rm --env-file .env orange-quant:latest --once --model models/binance20_lgb.pkl

# 停止
docker compose down
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--hour` | 0 | 调仓 UTC 小时 |
| `--minute` | 15 | 调仓 UTC 分钟 |
| `--topk` | 5 | 持仓数量 |
| `--dry-run` | - | 只分析不下单 |
| `--once` | - | 执行一次退出 |
| `--model` | models/binance20_lgb.pkl | LightGBM 模型路径 |

## 本地测试

```bash
source .venv/bin/activate
python scripts/run_binance_testnet.py --dry-run     # 分析
python scripts/run_binance_testnet.py --trade       # 实际下单
python scripts/server_entrypoint.py --once --dry-run # 单次
```

## 当前持仓

```bash
source .venv/bin/activate
python -c "
from dotenv import load_dotenv; load_dotenv(override=True)
from orange_quant.trading.broker import BinanceBroker
broker = BinanceBroker(testnet=False, paper=False)
balances = broker.get_balances()
for a, amt in sorted(balances.items()):
    if amt > 0.0001:
        p = broker.get_current_prices([f'{a}/USDT']).get(f'{a}/USDT', 0) if a != 'USDT' else 1
        print(f'  {a}: {amt:.6f} (≈\${amt*p:,.2f})')
"
```

## 安全提醒

- 默认每日 08:15 北京时间调仓
- 市价单，立即成交
- 最小交易金额 $20 USDT（低于此值不交易）
- 单币种最大仓位 25%
- TRX 等小数位多的币种可能有 dust 残留（自动跳过）
