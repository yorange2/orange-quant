FROM python:3.11-slim

LABEL description="Orange Quant Auto Trading Server"

WORKDIR /app

# LightGBM 需要 libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖（pyqlib 从 PyPI）
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# 复制项目代码
COPY orange_quant/ ./orange_quant/
COPY scripts/biance/ ./scripts/biance/

# 健康检查
HEALTHCHECK --interval=6h --timeout=30s --retries=3 \
    CMD python -c "import ccxt; ccxt.binance().load_markets()" || exit 1

# 每日 UTC 00:15 调仓
ENTRYPOINT ["python", "scripts/biance/execute.py"]
CMD ["--hour", "0", "--minute", "15"]
