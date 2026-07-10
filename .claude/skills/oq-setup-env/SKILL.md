---
name: oq-setup-env
description: Configure the orange-quant development/runtime environment and install all dependencies (macOS / Windows / Linux)
---

# Setup Environment

Configure the orange-quant development/runtime environment and install all dependencies. Supports macOS, Windows, and Linux.

## Trigger conditions
- "setup environment" / "install dependencies" / "setup"
- After first cloning the project

---

## Steps

### 1. Detect the platform

First confirm the current OS; subsequent steps pick the matching commands per platform:

| Platform | Identifier |
|------|------|
| macOS | `uname` = Darwin |
| Windows | `$env:OS` = Windows_NT |
| Linux | `uname` = Linux |

### 2. Install system dependencies

#### macOS

```bash
# Xcode Command Line Tools (provides the Clang compiler)
xcode-select --install 2>/dev/null || echo "already installed"

# OpenMP required by LightGBM
brew install libomp
```

#### Windows

```powershell
# Install Python (3.9+)
winget install Python.Python.3.11 --silent

# Install Git (if not present)
winget install Git.Git --silent
```

> On Windows, LightGBM's pip wheel bundles the OpenMP DLL, so no extra system package is needed.
> To build from source, install Visual Studio Build Tools (with the C++ workload).

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git build-essential

# OpenMP required by LightGBM (usually bundled, but ensure it's installed)
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

### 3. Create a virtual environment

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

### 4. Install Python dependencies

Same across all platforms:

```bash
pip install --upgrade pip
pip install git+https://github.com/microsoft/qlib.git
pip install lightgbm pandas numpy pyyaml ccxt python-dotenv jupyter ipykernel
```

> On Windows, if `lightgbm` fails to install, try `pip install lightgbm --only-binary=:all:` to use a prebuilt wheel.

### 5. Install ipykernel (optional, for Jupyter)

```bash
python -m ipykernel install --user --name=orange-quant --display-name="Orange Quant"
```

### 6. Install Docker (optional, needed for live trading)

- **macOS**: [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- **Windows**: [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- **Linux**: `curl -fsSL https://get.docker.com | sudo sh` then `sudo usermod -aG docker $USER`

### 7. Create the .env file

```bash
echo 'BINANCE_API_KEY=your_api_key_here' > .env
echo 'BIANCE_SECRET_KEY=your_secret_key_here' >> .env
```

> Live trading requires real API key/secret values.

### 8. Verify

```bash
python -c "import qlib; import lightgbm; import ccxt; import pandas; import numpy; print('OK: all packages loaded')"
```

Expected output: `OK: all packages loaded`.

---

## Platform differences at a glance

| Dependency | macOS | Windows | Linux (Debian) | Linux (RHEL) |
|------|-------|---------|----------------|--------------|
| Python 3.9+ | bundled with the OS | `winget install` | `apt install python3` | `dnf install python3` |
| OpenMP | `brew install libomp` | bundled in the pip wheel | `apt install libomp-dev` | `dnf install libomp-devel` |
| Compiler | `xcode-select --install` | VS Build Tools | `apt install build-essential` | `dnf install gcc gcc-c++` |
| Docker | Docker Desktop | Docker Desktop | `get.docker.com` | `get.docker.com` |
| Activate venv | `source .venv/bin/activate` | `.venv\Scripts\activate` | same as macOS | same as macOS |

## Notes

- Python version: 3.9+ (required by qlib)
- Install qlib from the GitHub source (the `pyqlib` package on PyPI is an older version and may have limited functionality in Docker)
- The `.env` file contains API keys, is already in `.gitignore`, and won't be committed
- If you hit SSL certificate issues (intranet/minimal Linux installs), install the `ca-certificates` package
