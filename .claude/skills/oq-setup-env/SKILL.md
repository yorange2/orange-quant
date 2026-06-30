---
name: oq-setup-env
description: 配置 orange-quant 开发/运行环境，安装所有依赖（macOS / Windows / Linux）
---

# Setup Environment

配置 orange-quant 开发/运行环境，安装所有依赖。支持 macOS、Windows、Linux 三平台。

## 触发条件
- "配置环境" / "安装依赖" / "setup"
- 首次克隆项目后

---

## 步骤

### 1. 检测平台

先确认当前操作系统，后续步骤按平台选择对应命令：

| 平台 | 标识 |
|------|------|
| macOS | `uname` = Darwin |
| Windows | `$env:OS` = Windows_NT |
| Linux | `uname` = Linux |

### 2. 安装系统依赖

#### macOS

```bash
# Xcode Command Line Tools（提供 Clang 编译器）
xcode-select --install 2>/dev/null || echo "已安装"

# LightGBM 需要的 OpenMP
brew install libomp
```

#### Windows

```powershell
# 安装 Python（3.9+）
winget install Python.Python.3.11 --silent

# 安装 Git（如果没有）
winget install Git.Git --silent
```

> Windows 上 LightGBM 的 pip wheel 已内置 OpenMP DLL，无需额外安装系统包。
> 如需编译源码，需安装 Visual Studio Build Tools（含 C++ 工作负载）。

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git build-essential

# LightGBM 需要的 OpenMP（通常已内置，确保安装）
sudo apt-get install -y libomp-dev
```

#### Linux (RHEL/CentOS/Fedora)

```bash
# RHEL/CentOS 8+
sudo dnf install -y python3 python3-pip git gcc gcc-c++ make
sudo dnf install -y libomp-devel

# CentOS 7
sudo yum install -y python3 python3-pip git gcc gcc-c++ make
sudo yum install -y libomp-devel
```

### 3. 创建虚拟环境

**macOS / Linux:**
```bash
cd /path/to/orange-quant
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
cd C:\path\to\orange-quant
python -m venv .venv
.venv\Scripts\activate
```

**Windows (CMD):**
```cmd
cd C:\path\to\orange-quant
python -m venv .venv
.venv\Scripts\activate.bat
```

### 4. 安装 Python 依赖

所有平台统一：

```bash
pip install --upgrade pip
pip install git+https://github.com/microsoft/qlib.git
pip install lightgbm pandas numpy pyyaml ccxt python-dotenv jupyter ipykernel
```

> Windows 若 `lightgbm` 安装报错，尝试 `pip install lightgbm --only-binary=:all:` 使用预编译 wheel。

### 5. 安装 ipykernel（可选，用于 Jupyter）

```bash
python -m ipykernel install --user --name=orange-quant --display-name="Orange Quant"
```

### 6. 安装 Docker（可选，实盘交易需要）

- **macOS**: [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- **Windows**: [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- **Linux**: `curl -fsSL https://get.docker.com | sudo sh` 然后 `sudo usermod -aG docker $USER`

### 7. 创建 .env 文件

```bash
echo 'BINANCE_API_KEY=your_api_key_here' > .env
echo 'BIANCE_SECRET_KEY=your_secret_key_here' >> .env
```

> 实盘交易必须填写真实的 API Key/Secret。

### 8. 验证

```bash
python -c "import qlib; import lightgbm; import ccxt; import pandas; import numpy; print('OK: all packages loaded')"
```

期望输出 `OK: all packages loaded`。

---

## 平台差异速查

| 依赖 | macOS | Windows | Linux (Debian) | Linux (RHEL) |
|------|-------|---------|----------------|--------------|
| Python 3.9+ | 系统自带 | `winget install` | `apt install python3` | `dnf install python3` |
| OpenMP | `brew install libomp` | pip wheel 内置 | `apt install libomp-dev` | `dnf install libomp-devel` |
| 编译器 | `xcode-select --install` | VS Build Tools | `apt install build-essential` | `dnf install gcc gcc-c++` |
| Docker | Docker Desktop | Docker Desktop | `get.docker.com` | `get.docker.com` |
| 虚拟环境激活 | `source .venv/bin/activate` | `.venv\Scripts\activate` | 同 macOS | 同 macOS |

## 注意事项

- Python 版本: 3.9+（qlib 要求）
- qlib 从 GitHub 源码安装（PyPI 的 `pyqlib` 是旧版本，可能在 Docker 中有功能限制）
- `.env` 文件包含 API 密钥，已加入 `.gitignore`，不会被提交
- 如遇到 SSL 证书问题（内网/Linux 最小安装），安装 `ca-certificates` 包
