---
name: oq-lgb-ab
description: Run an LGB A/B experiment — generate variant configs from a base, run dataset→train→backtest for each, produce the comparison table and a stability verdict (sub-window + bootstrap), then record the outcome in ROADMAP
argument-hint: "[base config] [variant spec...]"
---

# LGB A/B 实验（A 股 / crypto 横截面）

把一次 A/B 从"改配置 → 跑全链 → 判读"标准化，保证与其他实验可比。

## Trigger conditions
- "跑 A/B" / "对比实验" / "验证 XX 是否有效" / "换 top_n / cs_norm / loss / 钟点试试"
- 任何要在 `orange_quant.lgb` 上做的对比实验

## 工作流

### 1. 变体配置（只改一个变量，其余与基线逐字节一致）

- 改 knobs 后用 `scripts/gen_cn_ab_configs.py`（A 股：宽度/cs_norm/loss/行业中性）
  或 `scripts/gen_binance_hour_ab.py`（crypto 钟点），或手写
  `config/generated/<name>-<variant>.yaml`（gitignored，`load_config` 有 fallback）。
- **路径必须各自独立**：`model_dir/output_dir/cache_dir` 加后缀——npz 缓存不
  感知配置变化，同目录会静默复用错数据；换 universe/cs_norm/label 等关键参数后
  用 `--force` 重建。
- 可改的 knob（`orange_quant.lgb` 全配置化）：
  `universe.top_n` / `universe.min_amount`（流动性下限）/ `features.cs_norm`
  （rank|zscore|none）/ `label.industry_neutral` / `lgb.loss`
  （mse|lambdarank，后者另配 `ndcg_eval_at`）/ `data.hour_of_day`（crypto h1）。

### 2. 跑全链（每变体）

```bash
cd orange-quant
../.venv/bin/python -m orange_quant.lgb.dataset  <config>   # 大池子/重建时后台跑
../.venv/bin/python -m orange_quant.lgb.train     <config>
../.venv/bin/python -m orange_quant.lgb.backtest  <config>  # 自动带出 report.md + positions.csv
```

### 3. 对比表（两套口径都要看，缺一不可）

- **纯 alpha 层**：`outputs/<config>/report.md` 的 IC/ICIR、分年度 IC、decile
  阶梯、多空 spread t 统计量；
- **组合层**：metrics.json 的净超额（年化）、IR、换手、MDD。
- **教训（本仓库反复验证）**：IC/ICIR 与净超额会背离（top-800、csrank、
  indneutral、crypto zscore 四次出现）——IC 升但超额崩的改动不要采用；
  超额升但 alpha 层无证据（spread t 低）的同样不采用。

### 4. 稳定性判读（winner 必须过这一关）

```bash
../.venv/bin/python scripts/backtest_stability.py <基线> <变体>...
```

输出三向判读：
- 子窗口（默认 3 段）IC/超额是否全正；
- 跨配置排名是否跨窗口稳定（Spearman）；
- 全窗 winner 是否在每个窗口仍赢（不赢 = 单段行情驱动，判噪声）。

判级规则：全正 + 排名稳 → 结构可信；全正但排名乱 → "普遍弱正"可信、
"择优"不可信；任一窗口为负/翻转 → 全窗结论是 regime 噪声，不采用。

### 5. 记录（不记录 = 没做）

- 结果写回 `docs/ROADMAP.md`（勾选项/研究记录段），数字要能复现（test 区间、
  池子、config 名）；
- 基建改动（dataset/backtest/report/新脚本）走 PR（英文 commit，PR body 附
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`）；
- 负面结果照实记录（C5 lambdarank、C7 zscore 反哺均如此）——负面也是对照基线。

## 已测 knobs 结论速查（test 2025-2026，A 股 top-300 池）

| knob | 结论 | 状态 |
|---|---|---|
| `universe.top_n` 300→2000 | IC/ICIR 升，超额 edge 由单段行情撑 | top-300 组合层最稳 |
| `features.cs_norm: zscore` | IC 3/3 子窗 ≥ 基线，温和稳定正 | **建议默认基线** |
| `features.cs_norm: rank` | 超额崩 | 不采用 |
| `lgb.loss: lambdarank` | RankIC 转负 | 不采用 |
| `label.industry_neutral` | RankICIR 大升但超额崩 | 不采用 |
| crypto `hour_of_day` | 钟点全是正 IC，差异是噪声 | 不择优，可平均 |
