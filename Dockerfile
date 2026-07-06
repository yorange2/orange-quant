FROM python:3.11-slim

LABEL description="Orange Quant Auto Trading Server"

ENV MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

# LightGBM 需要 libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖（pyqlib 从 PyPI）
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# 复制项目代码
COPY biance_lgb_momtopk/ ./biance_lgb_momtopk/
COPY config/ ./config/

# 初始化空 git 仓库，消除 qlib recorder 的 git diff 警告
RUN git init && git config user.email "docker@orange-quant" && git config user.name "Docker"

# 健康检查
HEALTHCHECK --interval=6h --timeout=30s --retries=3 \
    CMD python -c "import ccxt; ccxt.binance().load_markets()" || exit 1

# 每日 UTC 00:15 调仓
ENTRYPOINT ["python", "-m", "biance_lgb_momtopk.server"]
CMD ["--hour", "0", "--minute", "15"]
