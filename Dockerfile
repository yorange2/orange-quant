FROM python:3.12-slim

LABEL description="Orange Quant RL Trading Server"

ENV MLFLOW_ALLOW_FILE_STORE=true \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy project code and configs
COPY orange_quant/ ./orange_quant/
COPY config/ ./config/

# Health check: liveness of the trading loop (stale heartbeat => unhealthy).
# start-period covers first boot + first run.
HEALTHCHECK --interval=5m --timeout=10s --start-period=15m --retries=3 \
    CMD python -m orange_quant.healthcheck || exit 1

# Rebalance daily at 00:15 UTC; override --config per venue
ENTRYPOINT ["python", "-m", "orange_quant.server"]
CMD ["--config", "binance-rl-rotation", "--hour", "0", "--minute", "15"]
