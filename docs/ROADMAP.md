# Orange Quant Roadmap：目录重构 + qlib MPS 支持

> 创建于 2026-08-08。基于当前代码库（qlib 0.9.8.dev32 本地源码、orange-quant 0.1.0）的实测分析。

## 1. 背景与目标

- **orange-quant**：核心框架（`orange_quant/`）+ 两个交易所适配器包（`biance_lgb_momtopk/`、`hyperliquid_lgb_momtopk/`）。适配器包结构高度镜像，存在真实重复（数据构建、broker）和大量 re-export 壳。
- **qlib MPS**：qlib 的 PyTorch 模型设备选择是 CUDA-or-CPU 硬编码；orange-quant 目前用运行时 workaround（`_prefer_mps` + `_cast_handler_float32`，`orange_quant/experiment.py:65,120`）在 Mac 上启用 MPS。目标是**在 qlib 源码层面原生支持 MPS**，让 `GPU="mps"` 开箱可用，随后移除 workaround。

### 目标
1. 目录结构简化：消除两个交易所包间的重复代码，保留 import 兼容性（config、skills、Dockerfile 均引用现有包名）。
2. qlib 原生 MPS：所有 PyTorch 模型（LSTM/GRU/Transformer/GATS/TabNet 等）支持 `GPU="mps"`，Apple Silicon 上训练可用且性能不劣于 CPU。
3. orange-quant 侧移除 MPS workaround，改由配置驱动。

---

## 2. 现状分析

### 2.1 orange-quant 目录现状（约 4600 行 Python）

```
orange_quant/                    # 核心（exchange-agnostic）— 已有共享管线
├── experiment.py       613 行   # 实验管线 + MPS workaround（_prefer_mps / _cast_handler_float32）
├── data/hourly.py      521 行   # 小时线下载 + phase 重采样（共享）
├── data/pipeline.py    259 行   # qlib 数据构建管线（共享，DataSource hooks 抽象）
├── runner.py           426 行   # 策略执行器（共享）
├── server.py           337 行   # 交易服务入口
├── model_predictor.py  132 行   # 预测器（共享）
├── trading/paper_broker.py 146  # 模拟券商
├── ensemble.py / train.py / spec.py / blacklist.py / healthcheck.py
│
biance_lgb_momtopk/              # Binance 适配器（⚠️ 包名拼写，见 §5）
├── data/build.py       193 行   # 真实重复：DataSource hooks（fetch 逻辑）
├── trading/broker.py   154 行   # 真实实现：BinanceBroker
├── trading/{runner,model_predictor,blacklist}.py  # 各 5 行，纯 re-export 壳
├── server.py / spec.py / train.py / workflow/experiment.py  # 壳
│
hyperliquid_lgb_momtopk/         # Hyperliquid 适配器（结构同左）
├── data/build.py       220 行   # 真实重复（与 binance 版 diff 达 305 行差异）
├── trading/broker.py   183 行   # HyperliquidBroker
├── trading/{runner,model_predictor}.py  # 纯 re-export 壳
│
scripts/                         # update_cn_data(240) merge_cn_update(144) strategy_sweep(366) phase_study(198) csi300/build_data(47)
config/                          # 9 个实验 yaml（csi300/binance/hyperliquid × lgb/lstm/gru/transformer）
data/                            # 软链接 → ~/.qlib/qlib_data/cn_data
models/                          # 导出的 pkl
.claude/skills/                  # oq-live-trade / oq-train-backtest / oq-download-data / oq-setup-env
```

**重复点**：
| 文件 | binance | hyperliquid | 差异实质 |
|---|---|---|---|
| `data/build.py` | 193 行 | 220 行 | 仅 fetch hooks（top-symbol 排名、API 端点、原始路径）不同，下载/构建循环已共享 |
| `trading/broker.py` | 154 行 | 183 行 | 接口对齐但实现独立（ccxt binance/hyperliquid）|
| `trading/{runner,model_predictor,blacklist}.py` | 5 行 | 5 行 | 纯 re-export（早前已迁移到 orange_quant）|
| `server.py` / `spec.py` / `train.py` / `workflow/` | 壳 | 壳 | — |

### 2.2 qlib MPS 现状

| 位置 | 现状 | MPS 适配点 |
|---|---|---|
| `qlib/contrib/model/pytorch_nn.py:93-96`（基类） | `GPU` 支持字符串：`torch.device(GPU)` → `GPU="mps"` 理论可用 | ① `torch.cuda.empty_cache()`（:336）未条件化；② `DataParallel`（:135）在 MPS 不可用 |
| `pytorch_lstm.py:72` / `pytorch_gru.py:73` / `pytorch_transformer.py:60` / `pytorch_gats.py:78` 等 6+ 个模型 | **硬编码** `cuda:%d if cuda else cpu`，未复用基类逻辑 | ③ 统一设备选择逻辑（加 mps 分支）|
| 数据流 | qlib 喂 float64 batch，MPS 拒绝 float64 | ④ fit/predict 前 downcast float32（orange_quant 已在 handler 层 workaround）|
| `qlib/rl/` | `auto_device` 跟随参数设备 | 基本无需改（验证即可）|
| 测试 | 无 MPS 专项测试 | ⑤ 新增 `GPU="mps"` 冒烟测试 |

**关键事实**：orange_quant 的 `_prefer_mps`（`experiment.py:65`）已证明 MPS 训练可用，但依赖 `PYTORCH_ENABLE_MPS_FALLBACK=1` 且小模型上 device 传输可能比 CPU 慢（有 `ORANGE_DISABLE_MPS` 逃生口）。qlib 原生化后应解决/文档化这两个问题。

---

## 3. Phase 1：qlib 原生 MPS 支持（优先，工作量约 3-4 天）

> 全部改动在 `qlib/` fork（`yorange2/qlib`）上完成，可开 PR 回馈上游。

### P1.1 设备选择统一（1-1.5 天）
- `pytorch_nn.py`：`GPU` 解析逻辑抽为辅助函数（`resolve_device(GPU)`），支持 `int`（cuda 序号）、`"cpu"`、`"cuda[:N]"`、`"mps"`。
- 各模型文件（lstm/gru/transformer/gats/tabnet/adarnn/localformer）的设备逻辑改为复用同一函数；如无法复用则补 mps 分支。
- `use_gpu` 属性（`device != cpu`）语义保持兼容（mps 算 True）。

### P1.2 MPS 运行时保护（0.5 天）
- `torch.cuda.empty_cache()` → `if torch.cuda.is_available(): ...`。
- `DataParallel` 仅在 `cuda` 设备启用（mps/cpu 时忽略 `data_parall=True` 并告警）。

### P1.3 float32 数据流（0.5 天）
- fit/predict 中 batch `.to(device)` 前统一 `.float()`（或确认 qlib handler 输出已是 float32）。
- 检查 `qlib/data/dataset/__init__.py` 采样输出 dtype，必要时在 loader 层转换。

### P1.4 验证与基准（1-1.5 天）
- 冒烟：LSTM/GRU/Transformer 用 `GPU="mps"` 训练 50 步不报错、loss 下降。
- 一致性：同 seed 下 MPS 与 CPU 训练结果可比（IC 差异 < 0.005）。
- 性能：MPS vs CPU 训练耗时基准（`examples/benchmarks` 的 LSTM 配置）；若 MPS 慢于 CPU（小模型常见），记录并文档化建议（大 batch / 大模型才值得用 MPS）。
- 单测：`tests/` 新增 mps 冒烟（`skipUnless(torch.backends.mps.is_available())`）。

**交付**：qlib fork 上 1-2 个 commit + 可选上游 PR；`GPU="mps"` 在 orange-quant 配置直接可用。

---

## 4. Phase 2：orange-quant 目录重构（约 2-3 天）

> 原则：**不破坏 import 兼容**——config yaml、`.claude/skills/`、Dockerfile 引用 `biance_lgb_momtopk.*` 与 `hyperliquid_lgb_momtopk.*` 的路径全部保留可用。

### R2.1 Broker 接口统一（0.5-1 天）
- 新增 `orange_quant/trading/broker.py`：定义 `Broker` ABC（`get_balances` / `get_current_prices` / `fetch_ohlcv` / `get_quote_volumes` / `get_min_notional` / `place_order`…，按两个现有实现的实际交集归纳）。
- `BinanceBroker` / `HyperliquidBroker` 改为继承 ABC；差异方法（如 `get_usdc_balance`）保留在各实现。
- 两个包的 `trading/broker.py` 变为 re-export（对齐现有壳模式）。

### R2.2 数据构建参数化（0.5-1 天）
- 将 `biance_lgb_momtopk/data/build.py` 与 `hyperliquid_lgb_momtopk/data/build.py` 的 fetch hooks 提炼为 `orange_quant/data/sources.py`（`BinanceSource` / `HyperliquidSource`，实现 `get_top_symbols` + `fetch_daily`）。
- 构建入口统一为 `python -m orange_quant.data.build --exchange binance|hyperliquid --top 50`。
- 保留 `python -m biance_lgb_momtopk.data.build` 作为兼容入口（转发）。

### R2.3 目录瘦身（0.5 天）
- 删除两包内纯壳文件（`trading/runner.py` 等若已无人直接引用则删；有引用则保留 re-export）。
- `server.py` / `spec.py` / `train.py` 按 R2.1/R2.2 结果压缩。
- **决策点**（实施前与用户确认）：是否将两包收敛为 `orange_quant/exchanges/{binance,hyperliquid}/`？推荐**保留现有包名**（最小破坏面），仅在包内去重。

### R2.4 文档与技能同步（0.5 天）
- `.claude/skills/` 4 个 skill 的路径引用核对更新。
- `README.md` 项目结构段落更新。
- `pyproject.toml` 包发现规则更新（如新增 `orange_quant.exchanges`）。

---

## 5. Phase 3：MPS workaround 移除与配置化（约 0.5-1 天，依赖 Phase 1 完成）

- `orange_quant/experiment.py`：删除 `_prefer_mps` / `_cast_handler_float32` 及调用点（:539/:544）。
- DL 配置（`config/*-lstm/gru/transformer.yaml`）的 model kwargs 增加 `GPU: mps`（或按 skill 默认注入），由 qlib 原生处理。
- `ORANGE_DISABLE_MPS` 逃生口移除（qlib 原生支持后可用 `GPU: cpu` 等价表达）。
- 验证 `csi300-lstm-momtopk` 在 MPS 上端到端跑通。

---

## 6. 可选后续（低优先级）

- **O1** 修正 `biance_lgb_momtopk` → `binance_lgb_momtopk` 拼写：破坏所有 import/config/skill 引用，需一次性 grep 迁移 + 兼容 shim；收益仅为命名整洁。建议单独评估。
- **O2** 向 qlib 上游提 PR（P1 完成后），若被合并则后续可从 PyPI 安装，不必 fork。
- **O3** 数据更新脚本（`scripts/update_cn_data.py` / `merge_cn_update.py`）移入 `orange_quant/data/` 统一管理。

---

## 7. 工作量汇总

| 阶段 | 内容 | 工作量 |
|---|---|---|
| P1.1 | qlib 设备选择统一 | 1-1.5 天 |
| P1.2 | MPS 运行时保护（empty_cache/DataParallel） | 0.5 天 |
| P1.3 | float32 数据流 | 0.5 天 |
| P1.4 | MPS 验证与基准 | 1-1.5 天 |
| **Phase 1 小计** | **qlib MPS 原生支持** | **3-4 天** |
| R2.1 | Broker ABC 统一 | 0.5-1 天 |
| R2.2 | 数据构建参数化 | 0.5-1 天 |
| R2.3 | 目录瘦身 | 0.5 天 |
| R2.4 | 文档/技能同步 | 0.5 天 |
| **Phase 2 小计** | **orange-quant 重构** | **2-3 天** |
| P3 | workaround 移除与配置化 | 0.5-1 天 |
| **合计** | | **约 6-8 天** |

## 8. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| MPS 上 LSTM/GRU 部分算子缺失（fallback 到 CPU 逐算子传输） | 性能可能不升反降 | P1.4 先做基准再决定默认启用；文档化 `GPU: cpu` 建议 |
| 重构破坏现有 import（skills/Dockerfile 引用包路径） | 部署中断 | R2 全程保留 re-export 壳；每步跑 `python -c "import ..."` 冒烟 + 跑一次 `csi300-lgb-momtopk` 回归 |
| qlib fork 与上游 main 分叉 | 后续同步成本 | P1 改动集中且小；保持提交可独立成 PR |
| float32 转换影响 CPU/CUDA 训练结果 | 模型结果漂移 | P1.3 仅限 MPS 路径或与现有 CPU 结果做回归对比 |

## 8.1 ⚠️ 已发现 Blockers（2026-08-08）

### Blocker-1：qlib PyTorch 模型 fit 在本环境段错误（P1.4 受阻）

**现象**：`DNNModelPytorch.fit` 在 macOS arm64 + Python 3.12.13 + numpy 2.5.1 + pandas 2.3.3 + joblib 1.5.3 环境下**必然段错误**（SIGSEGV）或 GIL 死锁，与 torch 版本无关（2.5.1 / 2.7.1 / 2.13.0 三版均复现），**原版 qlib（未修改）同样崩溃**。

**已定位的代码缺陷**（qlib 上游）：`pytorch_nn.py` fit 中
```python
all_t[v][seg] = torch.from_numpy(all_df[v][seg].values).float()
...
del df; del all_df["x"]; gc.collect()
```
Alpha158 输出 float32 → `from_numpy().float()` **同 dtype 不复制**（共享内存）→ `del + gc.collect()` 后张量悬垂 → 后续访问段错误。已用 `torch.tensor(..., dtype=float32)`（总是复制）修复**该缺陷**，但修复后仍崩（崩溃点移动到 tensor 索引/构建）→ 存在**第二层环境级不兼容**（疑 numpy 2.5 的 buffer 释放语义或 torch 2.x 与 qlib Cython 扩展的交互）。

**影响**：P1.4（MPS 训练验证与基准）无法在当前环境执行；MPS 支持代码（resolve_device/DataParallel 保护/empty_cache 保护/悬垂修复）已完成但**未经运行验证**。

**✅ 已解决（2026-08-09）**。根因有三层，全部定位并修复：

1. **libomp/OpenMP 多运行时冲突**（环境级）：崩溃在 `libomp.dylib __kmp_suspend_64`（EXC_BAD_ACCESS）。torch 自带 libomp 与 homebrew libomp 冲突。**绕过：`OMP_NUM_THREADS=1`**。
2. **NaN 特征污染 BatchNorm**（qlib 上游缺陷）：Alpha158 特征 ~1% NaN → BatchNorm batch 统计变 NaN → 整个 batch 输出 NaN → loss NaN → best checkpoint 永不保存（`EOFError`）。**修复：fit/predict 路径 `torch.nan_to_num` 填 0 + `get_loss` 加 NaN mask**（已合并 PR #2）。
3. **悬垂张量**（qlib 上游缺陷，PR #1 已修）。

**验证结果**（python 3.11 venv：`.venv311`，numpy 2.4.6 + pandas 2.3.3 + torch 2.5.1 + qlib dev35）：
- `DNNModelPytorch.fit` 完整跑通（CPU 与 `GPU="mps"` 均可）
- CPU vs MPS 预测 Pearson 相关 **0.9455**（同 seed）
- **性能基准：小 DNN 模型 MPS 11.0s vs CPU 2.0s（MPS 慢 5.5x）**——roadmap §8 预判兑现：小模型 MPS 不划算，默认建议 CPU；大模型/大 batch 才值得 MPS

**环境要求**：py3.11（`.venv311`）+ `OMP_NUM_THREADS=1` + `MLFLOW_ALLOW_FILE_STORE=true`。py3.12 环境（`.venv`）未解阻（第 2/3 层修复已在，第 1 层 OMP 变量同样适用，可复测）。

**当前状态**：P1.1-P1.4 完成（代码就绪、25 个模型文件统一 resolve_device、导入验证通过、MPS 冒烟单测已合并 PR #3）；P3 完成（workaround 移除、`GPU: mps` 配置化、csi300-lstm-momtopk 端到端验证通过，orange-quant PR #15）。**R2 重构（orange-quant 目录）不受影响**（LightGBM 管线正常）。

### 2026-08-09 执行记录

- **P1.4 ✅**：MPS 冒烟单测固化进 `qlib/tests/model/test_pytorch_mps.py`（`skipUnless(torch.backends.mps.is_available())`）。LSTM/GRU/TransformerModel 以 `GPU="mps"` 通过合成数据 stub 驱动公开 `fit`/`predict`，断言 device 解析为 mps、loss 逐 epoch 下降、预测有限；已合并 fork PR **#3**（`yorange2/qlib`）。实测三模型 MPS 端到端训练 OK（`.venv311`、`OMP_NUM_THREADS=1`、`MLFLOW_ALLOW_FILE_STORE=true`）。
- 注：其余 23 处 `torch.cuda.empty_cache()` 仍是裸 `if self.use_gpu:` 守卫（MPS 下 use_gpu=True），macOS 构建上是 no-op 不影响 `GPU="mps"`，未随 PR #3 改动（保持 test-only）。
- **P1.3 补完（TS 变体 float32 数据流）**：端到端跑 `csi300-lstm-momtopk`（`GPU="mps"`）暴露两个真实数据路径问题，均已在 qlib fork 修复并合并：
  - PR **#4**：`*_ts` 模型默认样本权重 `np.ones(..., dtype=np.float32)`（原 float64，`weight.to(mps)` 直接抛错）。
  - PR **#5**：`*_ts` 模型 fit/predict 循环中 feature/label 在 `.to(device)` 前统一 `.float()`（真实 handler 输出是 float64，MPS 拒绝）。冒烟测试改用 float64 合成数据（镜像真实输出），无修复时精确复现 TypeError。
- **O3 ✅**：`scripts/update_cn_data.py`、`merge_cn_update.py` 移入 `orange_quant/data/`（`python -m orange_quant.data.update_cn_data`），scripts/ 保留转发 shim；已合并 orange-quant PR **#14**。
- **P3 ✅**：`experiment.py` 删除 `_prefer_mps` / `_cast_handler_float32` / `_float32_reweighter` 及调用点（-124 行），`ORANGE_DISABLE_MPS` 逃生口移除（`GPU: cpu` 等价表达）；5 个 DL config 的 model kwargs 改为 `GPU: mps`（CPU 机器用 `GPU: cpu`）。**端到端验证**（py3.12 `.venv` + `OMP_NUM_THREADS=1`）：`csi300-lstm-momtopk` 无 workaround 全链路跑通——`device: mps` 训练 10 epochs（early stop，train loss 0.953→0.941）、IC 0.034 / Rank IC 0.043、871 天回测（无成本年化 11.2%）、模型导出 `models/csi300-lstm-momtopk.pkl`；`pa/pos=0` 为已知 PortAnaRecord 报告 bug。已合并 orange-quant PR **#15**。基准结论（§8.1）：小模型 MPS 慢于 CPU（5.5x），大 batch/大模型才值得 MPS，配置注释已文档化。

### 2026-08-08 执行记录

- **P1.1-P1.3 ✅**：`resolve_device()` 统一 25 个模型文件的设备选择；`DataParallel` 仅 cuda 启用；`torch.cuda.empty_cache()` 条件化（23 处）；float32 数据流确认已完备。另修复 qlib 上游悬垂张量缺陷（`from_numpy().float()` 共享内存 + `del`/`gc` → 段错误），改用 `torch.tensor(..., dtype=float32)`。
- **P1.4 ⛔ blocked**：Blocker-1（见 §8.1）。
- **R2.1 ✅**：`orange_quant/trading/broker.py` 定义 `Broker` ABC（10 个抽象方法）；`BinanceBroker`/`HyperliquidBroker`/`PaperBroker` 均继承且全部实现。
- **R2.2 ✅**：`orange_quant/data/sources.py`（`BinanceSource`/`HyperliquidSource` hooks）+ 统一入口 `python -m orange_quant.data.build --exchange binance|hyperliquid`；旧入口保留薄 shim（兼容 `biance_lgb_momtopk.data.build` 等）。
- **R2.3 ✅**：壳文件全部压缩为必要 shim（包名被 Dockerfile/skills/config 引用，保留）；38 个模块全量导入冒烟通过。
- **R2.4 ✅**：README 结构段、oq-download-data skill 已同步。
- **P3 ⏸ 部分完成**：`_prefer_mps` 标注为幂等 fallback（qlib 原生 `GPU: mps` 已就绪），移除与验证等待 Blocker 解除。

## 9. 建议执行顺序

```
Phase 1 (qlib MPS) ──► Phase 3 (去 workaround，配置化)   ← 用户近期痛点：Mac 上跑 DL
Phase 2 (目录重构)  ──► 与 Phase 1 并行（不同仓库，互不依赖）
可选 O1/O2/O3 ──────► 收尾
```
