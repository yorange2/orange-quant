FROM python:3.11-slim

LABEL description="Orange Quant Auto Trading Server"

ENV MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

# LightGBM requires libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY orange_quant/ ./orange_quant/
COPY biance_lgb_momtopk/ ./biance_lgb_momtopk/
COPY hyperliquid_lgb_momtopk/ ./hyperliquid_lgb_momtopk/
COPY config/ ./config/

# Initialize an empty git repo to silence the qlib recorder's git diff warning
RUN git init && git config user.email "docker@orange-quant" && git config user.name "Docker"

# Health check: liveness of the trading loop (stale heartbeat => unhealthy),
# not just network connectivity. start-period covers first boot + initial retrain.
HEALTHCHECK --interval=5m --timeout=10s --start-period=15m --retries=3 \
    CMD python -m orange_quant.healthcheck || exit 1

# Rebalance daily at 00:15 UTC
ENTRYPOINT ["python", "-m", "biance_lgb_momtopk.server"]
CMD ["--hour", "0", "--minute", "15"]
