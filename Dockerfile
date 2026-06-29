FROM python:3.9-slim

LABEL description="Orange Quant Auto Trading Server"

WORKDIR /app

# 安装依赖
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# 复制项目代码（只复制交易相关模块）
COPY orange_quant/ ./orange_quant/
COPY scripts/server_entrypoint.py ./scripts/

# 健康检查
HEALTHCHECK --interval=6h --timeout=30s --retries=3 \
    CMD python -c "import ccxt; ccxt.binance().load_markets()" || exit 1

# 每日 UTC 00:15 调仓
ENTRYPOINT ["python", "scripts/server_entrypoint.py"]
CMD ["--hour", "0", "--minute", "15"]
