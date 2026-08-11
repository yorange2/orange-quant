# Orange Quant Roadmap：横截面选股（A 股 LGB 主线）

> 创建于 2026-08-11。上一版路线图（RL 数据扩充 R1–R6）已完成，见 `docs/finished/ROADMAP-rl-data.md`；更早的 qlib 重构版见 `docs/finished/ROADMAP.md`。
> 核心论点：**横截面宽度是最便宜的真实样本扩充**——每天 N 只股票同场竞技，N 从 50 → 2000 是几十倍的独立信息增量（对比时序重采样只是同一根 K 线的影子）。本路线图把 A 股 LGB 选股迁到新架构后，沿"加宽截面 → 强化评估 → 损失/中性化研究"推进。

## 现状（2026-08-11）

- 新架构 LGB 流水线（`orange_quant.lgb.{dataset,train,backtest}`）市场无关，但**只有币安配置**（`binance-lgb-momtopk`：IC 0.034 / 净超额 +17.3% / IR 0.77，test 2026-02~06）
- A 股 LGB 停在 legacy qlib 版（git tag `legacy-pre-rl-refactor`）：csi300-lgb-momtopk IC 0.035（test 2017-2020）；-2026 版（test 2023-2026）组合层指标因 qlib dev 版 PortAnaRecord bug 不可信，仅 IC 可信
- `data/cn_raw/` 已有 **4826 只**股票日线 CSV（腾讯源，复权+单位换算已在 `data/tencent.py` 处理，至 2026-08）
- `universe.freeze_universe` 支持 A 股流动性池（按日均成交额 top-N，冻结无前视）
- RL 联合训练（R4）已证明 A 股横截面对策略泛化有正贡献

## 路线图（按性价比排序，可独立落地）

### C1. A 股 LGB 基线迁移到新架构（纯工程）

**动机**：一切 A/B 的前提。legacy qlib 版报告层有 bug、股票池写死 csi300 成分，新架构回测自算不受影响。

- [x] `config/cn-lgb-momtopk.yaml`：`market.type: cn`，top-300 流动性池（`liquidity_start/freeze_date` 参照 RL 的 A 股配置），train/valid/test ≈ 2018-2023 / 2024 / 2025-2026-08
- [x] 跑通 dataset → train → backtest 全链路；确认 A 股 CSV（列名/停牌日/涨跌停日）在共享日历对齐逻辑下无异常（发现并修复：停牌日持仓估值 NaN 传染 → ffill 按最后一笔收盘价估值，交易仍要求真实 bar）
- [x] backtest 基准换成指数或等权（`benchmark` 对 A 股取 SH000300，确认 cn_raw 里有指数 CSV，没有则用等权池）
- [x] 记录基线三件套：**IC / ICIR / TopK 净超额**，作为后续所有 A/B 的对照 —— **IC 0.0453 / ICIR 0.255 / 净超额 +23.8% 年化**（test 2025-01~2026-07，top-300 池，vs SH000300，IR 0.675，换手 24.6 次/年；PR #21）

### C2. 截面加宽 A/B（top-50 → 300 → 800 → 2000+）

**动机**：横截面样本 ×N 是本路线图的主菜；且小市值票上因子通常更有效（拥挤度低）。

- [x] 同一配置只改 `universe.top_n`，四档各跑一次，对比 IC / ICIR / 净超额 / 换手 —— `scripts/gen_cn_ab_configs.py` 生成四档（PR #23）；test 2025-2026：**top-50 IC 0.0001 / top-300 0.0453 / top-800 0.0494 / top-2000 0.0570**；ICIR 0.000→0.592 严格随宽度上升；净超额 top-50 −8.9% / top-300 +23.8% / top-800 +12.5% / top-2000 +46.1%；换手 16~25 次/年
- [x] 加流动性下限过滤（日均成交额绝对门槛），剔除不可交易小票，避免"纸面 alpha" —— `freeze_universe` 新增 `min_amount`（cn 基线 5000 万元，只切 2000 名以下，四档同规则）；2017 冻结窗过 200 日门槛的池仅 2656 只
- [x] 关注两个方向的张力：截面越宽 IC 通常越高，但 TopK 落在小票上的**容量与冲击成本**越差——净超额与 IC 结论可能背离，两个都要看 —— 背离已确认：top-800 IC 0.0494 > top-300 但净超额 12.5% < 23.8%（组合层路径噪声）；top-2000 TopK 落 0.5 亿成交额小票，真实资金需重估容量（当前 100 万账户无冲击）
- [x] 若 2000+ 档训练内存吃紧，dataset 层 npz 缓存分段加载 —— **不需要**：top-2000 feats 2.8GB，32GB 内存下 dataset/train/backtest 全链顺畅（C2.4 关闭）

### C3. 横截面评估报告（研究基建）

**动机**：宽截面 A/B 只看单个 IC 数字会误判，需要一套标准报告。

- [x] `lgb/backtest.py`（或独立 `lgb/report.py`）输出：per-day IC 序列 + 累计图、ICIR、分年度 IC、**decile 收益单调性**（预测分 10 档的实际收益阶梯）—— 已实现 `lgb/report.py`，`backtest` 主流程自动生成（PR #22）
- [x] **多空 spread**（top decile − bottom decile 日收益）：纯 alpha 度量，剥离市场 beta——A 股实盘难做空，仅作评估指标，不进交易 —— 含 t 统计量；C1 基线上 spread 日均 +0.00237（t=3.62）
- [x] 报告落 `outputs/<config>/report.md` + png，方便跨实验对比 —— `report.md` / `report_ic.png` / `report_deciles.png`；metrics.json 补 `icir`/`rank_icir`

### C4. 特征截面标准化 A/B（研究）

**动机**：当前 raw Alpha158 直接喂 LGB。树模型对单调变换不敏感，但特征的**跨日分布漂移**会让分裂点在不同市况下含义不同；per-date 截面 rank 让"排前 10%"跨时期同义。

- [x] dataset 层加可选 `features.cs_norm: rank|zscore|none`（per-date，对非 NaN 截面）—— 已实现（PR #24），默认 none 保持 legacy 原始 Alpha158 字节不变
- [x] A/B vs C1 基线；预期增益温和（Alpha158 多为比率型特征），负面则记录后关闭 —— **zscore 温和正向，保留为可选**：IC 0.0453→0.0513、ICIR 0.255→0.295、decile spread t 3.62→5.04、净超额 +23.8%→+24.9%；**rank 记录后不启用**（ICIR 0.274 微升但净超额掉到 +13.3%，组合层噪声，与 C2 top-800 同款背离）；默认仍 none（一次只改一个变量）

### C5. 排序损失 LambdaRank（研究）

**动机**：TopK 只消费排序，MSE 优化的是回归误差——目标错配。LightGBM 原生支持 lambdarank。

- [x] label 转每日相对收益分档（如 5 档 quantile 整数 relevance），`group` = 交易日（`date_idx` 已有，dataset 层现成）—— 已实现（PR #25）
- [x] `lgb.loss: lambdarank` 配置化，对比 MSE 基线的 IC / 净超额 —— **结果显著为负**：valid IC 0.0096 vs 0.0257、test IC 0.0136 vs 0.0453、RankIC −0.0255（转负）、净超额 −3.4% vs +23.8%、decile t 0.97；已排查非机制 bug（预测方差正常但逐日相关 ≈0/负）——exp gain + NDCG@10 在此池/区间学不到正确排序；基建保留（loss 分派可配置）
- [x] 备注：lambdarank 对 top 的加权天然契合 TopK 策略，值得认真做；但 A 股单日 group 大（300~2000），训练耗时预计上升 —— 训练耗时实测**不高**（5 seeds ~3 分钟，早停 130-250 轮）；后续候选调参（如线性 label_gain、更多分档、NDCG@50、宽池 top-2000 上重试）留待再研究，本路线图内不展开

### C6. 行业中性化（需新数据）

**动机**：Alpha158 的动量/波动率因子隐含行业贝塔，行业内相对强弱才是更纯的个股 alpha；组合层也避免 TopK 挤在单一行业。

- [x] 行业分类数据：腾讯/akshare 申万一级行业快照，落 `data/cn_industry.csv`（当前快照近似历史，survivorship caveat 与 universe 的 membership 同款，文档记录）—— `scripts/fetch_cn_industry.py`（PR #26）：akshare 申万官网接口（东财/新浪/历史分类 SSL 均被限），5202 只 × 31 行业；top-300 池覆盖 291/300（未覆盖为退市/吸收合并）
- [x] label 行业内 z-score（替代全截面 z-score）A/B —— **排序度量大幅提升但净超额崩掉**：valid IC 0.0257→0.0439、RankICIR 0.101→0.381，但净超额 +23.8%→**−9.4%**、decile t 3.62→1.62——全截面模型隐含行业动量轮动（基线 TopK 有色 15.6%/计算机 9.4%），2025-26 该部分贡献主要收益；行业内 alpha 更稳但量级小；且 label 中性化不动特征层原始水平（indneutral TopK 漂到防御银行 22.8%）。**记录，不采用**；彻底中性化需特征层处理（超出本路线图）
- [x] 报告层加组合行业暴露统计（TopK 持仓的行业集中度）—— report.md 新增行业暴露段（日均权重/top-3/HHI），backtest 输出 positions.csv；无行业映射（crypto）时自动跳过

### C7. 反哺 crypto LGB（可选，搭车）

- [ ] C4/C5 中验证为正向的改动应用到 `binance-lgb-momtopk`（48 币池截面小是其短板，排序损失可能同样受益）

## 实施顺序建议

```
C1（基线迁移，工程）→ C2 + C3 一起做（宽截面 A/B 需要评估报告才能判断）
→ C4 / C5（研究，可并行，各自独立 A/B）→ C6（需新数据源）→ C7（搭车）
```

C2+C3 完成后重新评估：若宽截面 IC 显著提升但净超额不动，优先查交易容量/成本假设，再决定是否继续 C4-C6。

## 环境/代码备忘

- **cwd 遮蔽坑**：一律 `cd orange-quant/` 后运行（workspace 根的 `qlib/` 目录会遮蔽 pyqlib 包）
- 新架构入口：`../.venv/bin/python -m orange_quant.lgb.{dataset,train,backtest} <config>`
- A 股原始数据更新：`oq-download-data` skill（腾讯 K 线，断点续跑）；单位换算在 `orange_quant/data/tencent.py`（volume 手→股、amount 万元→元，STAR/指数例外已处理）
- legacy A 股基线参照：git tag `legacy-pre-rl-refactor` 的 `config/csi300-lgb-momtopk.yaml`
- qlib dev 版 PortAnaRecord 报告不可信（`backtest()` 返回结构变更）；新架构 backtest 自算指标，不受影响
- label 语义沿用 `close[t+2]/close[t+1]-1`（qlib Alpha158 默认，T+1 可执行）；A 股注意停牌日 NaN label 已被 drop、涨停买不进属于回测乐观偏差（C2 容量讨论时一并评估）
