# Download Datasets

下载量化实验所需的数据集。支持 A 股、Crypto (Hyperliquid)、Binance 三种数据源。

## 触发条件
- "下载数据" / "download data"
- "构建数据集" / "build dataset"

## 可用数据集

### A 股（qlib 官方数据，已下载可跳过）

```bash
source .venv/bin/activate
python scripts/download_data.py
```

数据存放: `~/.qlib/qlib_data/cn_data/`（约 1-2 GB）

### Crypto 合约（Hyperliquid 日线，187 币种）

```bash
source .venv/bin/activate
python scripts/build_crypto_data.py
```

数据存放: `data/qlib_data/crypto/`（合约）、`data/crypto_raw/`（原始 CSV）

### Binance 现货日线（Top 50 成交量）

```bash
source .venv/bin/activate
python scripts/build_binance_data.py --top 50
```

数据存放: `data/qlib_data/binance/`（qlib 格式）、`data/binance_raw/`（原始 CSV）

### Binance 时线（1h，实验性）

```bash
source .venv/bin/activate
python scripts/build_crypto_data_1h.py
```

数据存放: `data/qlib_data/crypto_1h/`

## 注意事项

- 首次下载需要联网，耗时取决于网络
- Crypto 数据从 Hyperliquid/Binance 公开 API 获取，无需 API key
- 数据目录在 `.gitignore` 中，不会被提交
