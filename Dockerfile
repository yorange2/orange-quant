FROM python:3.12-slim

LABEL description="Orange Quant RL Trading Server"

ENV MLFLOW_ALLOW_FILE_STORE=true \
    PYTHONUNBUFFERED=1

WORKDIR /app

# LightGBM links against the OpenMP runtime, which python:*-slim does not ship;
# without it `import lightgbm` dies with "libgomp.so.1: cannot open shared
# object file". The RL path never hit this because torch bundles its own copy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies. Host is a Mac — the container never gets a GPU, so
# pull the CPU-only torch first (the PyPI linux wheel bundles ~3GB of CUDA
# libs); pip keeps the pre-installed torch since pyproject requires >=2.5.
# The CPU index caps at 2.5.1 for linux-aarch64 (as of 2026-08).
COPY pyproject.toml .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Copy project code and configs
COPY orange_quant/ ./orange_quant/
COPY config/ ./config/
COPY scripts/__init__.py scripts/retrain_live.py ./scripts/

# Health check: liveness of the trading loop (stale heartbeat => unhealthy).
# start-period covers first boot + first run.
HEALTHCHECK --interval=5m --timeout=10s --start-period=15m --retries=3 \
    CMD python -m orange_quant.healthcheck || exit 1

# Rebalance daily at 00:15 UTC; override --config per venue
ENTRYPOINT ["python", "-m", "orange_quant.server"]
CMD ["--config", "binance-rl-rotation", "--hour", "0", "--minute", "15"]
