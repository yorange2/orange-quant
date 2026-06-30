# Setup Environment

配置 orange-quant 开发/运行环境，安装所有依赖。

## 触发条件
- "配置环境" / "安装依赖" / "setup"
- 首次克隆项目后

## 步骤

### 1. 创建虚拟环境

```bash
cd /Users/yuanchengcheng/Documents/GitHub/orange-quant
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装 Python 依赖

```bash
pip install --upgrade pip
pip install git+https://github.com/microsoft/qlib.git
pip install lightgbm pandas numpy pyyaml ccxt python-dotenv jupyter ipykernel
```

### 3. macOS: 安装 LightGBM 系统依赖

```bash
brew install libomp
```

### 4. 安装 ipykernel（可选，用于 Jupyter）

```bash
python -m ipykernel install --user --name=orange-quant --display-name="Orange Quant"
```

### 5. 验证

```bash
python -c "import qlib; import lightgbm; import ccxt; print('OK')"
```

## 注意事项

- Python 版本: 3.9+（macOS 自带 3.9.6 可用）
- qlib 从 GitHub 安装（PyPI 的 `pyqlib` 在 Docker 中有限制）
- `.env` 文件需手动创建（含 `BINANCE_API_KEY` 和 `BIANCE_SECRET_KEY`）
