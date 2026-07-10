# 🍊 Orange Quant

AI quantitative trading algorithm framework based on [Microsoft qlib](https://github.com/microsoft/qlib).

## Project structure

```
orange-quant/
├── biance_lgb_momtopk/               # Core package
│   ├── trading/                # Trading layer: Binance automated trading
│   └── workflow/               # Experiment management: YAML config driven
├── config/                     # Experiment config files
│   ├── csi300-lgb-momtopk.yaml
│   └── binance-lgb-momtopk.yaml
├── scripts/                    # Utility scripts
│   └── csi300/
│       └── build_data.py       # Download A-share data
├── biance_lgb_momtopk/
│   ├── data/
│   │   └── build.py            # Build the Binance dataset
│   ├── server.py               # Trading execution entry point
│   └── train.py                # Train the LightGBM model
├── Dockerfile
└── docker-compose.yml
```

## Quick start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/microsoft/qlib.git
pip install lightgbm pandas numpy pyyaml ccxt python-dotenv
# macOS: brew install libomp
```

### 2. Download data and run an experiment

```bash
# A-shares
python scripts/csi300/build_data.py
python -c "from biance_lgb_momtopk.workflow.experiment import run_from_yaml; run_from_yaml('config/csi300-lgb-momtopk.yaml')"

# Binance
python -m biance_lgb_momtopk.data.build --top 50
python -c "from biance_lgb_momtopk.workflow.experiment import run_from_yaml; run_from_yaml('config/binance-lgb-momtopk.yaml')"
```

## Experiment results

| Dataset | Universe | IC | Excess return (net of costs) | IR (net of costs) |
|--------|------|-----|-------------|-----------|
| A-share CSI300 | 820 | 0.027 | 1.0% | 0.11 |
| Binance top-20 blue chips | 20 | 0.034 | 17.3% | 0.77 |

## Automated trading

### Local

```bash
source .venv/bin/activate

# DRY RUN (analyze only, no orders placed)
python -m biance_lgb_momtopk.server --once --dry-run --model models/binance-lgb-momtopk.pkl

# Live trading
python -m biance_lgb_momtopk.server --once --model models/binance-lgb-momtopk.pkl

# View holdings
python -c "
from dotenv import load_dotenv; load_dotenv(override=True)
from biance_lgb_momtopk.trading.broker import BinanceBroker
broker = BinanceBroker(testnet=False, paper=False)
for a, amt in sorted(broker.get_balances().items()):
    if amt > 0.0001:
        print(f'  {a}: {amt:.6f}')
"
```

### Docker deployment

```bash
# Prepare .env (Binance API key)
cat > .env << EOF
BINANCE_API_KEY=your_api_key
BIANCE_SECRET_KEY=your_secret_key
EOF

# Download data + place the model file
python -m biance_lgb_momtopk.data.build --top 50
mkdir -p models
cp /path/to/binance-lgb-momtopk.pkl models/

# Upload to the server
scp -r models data/qlib_data/binance user@server:~/orange-quant/
# Or run build_data and training directly on the server to produce the model

# Start
docker compose --profile live up -d  --build    # live trading
docker compose --profile dry-run up -d  --build # observe only (no orders)
docker compose --profile once up --build        # run once manually (for testing)
docker logs -f orange-quant             # logs
docker compose down                     # stop
```

### Upgrading

```bash
git pull
docker compose up -d --build    # rebuild and replace the old container
```

### Parameters

| Parameter | Default | Description |
|------|--------|------|
| `--hour` | 0 | Rebalance time (UTC hour) |
| `--minute` | 15 | Rebalance time (minute) |
| `--topk` | 5 | Number of positions to hold |
| `--lookback` | 160 | Lookback window in days |
| `--model` | models/binance-lgb-momtopk.pkl | LightGBM model path; falls back to the momentum strategy if not specified |
| `--dry-run` | — | Analyze only, no orders placed |
| `--once` | — | Run once then exit |
| `--testnet` | — | Use the Binance testnet |
| `--retrain` | — | Refresh data and retrain the model before rebalancing |

## License

MIT
