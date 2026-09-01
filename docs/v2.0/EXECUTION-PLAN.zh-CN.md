# AutoKV-Skip v2.0 阶段 2–4 执行计划

状态：待执行

日期：2026-09-02

依据：[v2.0 设计文档](DESIGN.zh-CN.md)

## 1. 本计划的范围

本文只展开三个阶段：

- 阶段 2：设计并冻结 Quality v2；
- 阶段 3：运行端点并执行质量缺口决策；
- 阶段 4：只在有缺口时搜索预算与层集合，并做 held-out 验证。

性能复验属于后续阶段 5，不在本文实施。阶段 2–4 的交付结果是一个经 held-out 验证的策略 `P*`，或一条可审计的 `P_0`/`P_32` 端点结论。

## 2. 前置条件

以下事项是阶段 2 的入口条件，不另建一串 gate：

- [ ] 按 [v1.0 对应源码发布要求](../v1.0/SOURCE-PUBLICATION-REQUIREMENT.zh-CN.md) 发布 `2c6a4e1...`，然后从该真实修复版源码开始开发 v2.0；
- [ ] v2.0 开发提交已推送到 GitHub，实验服务器检出的 commit 可公开访问，工作区干净；
- [ ] 实验服务器仍能启动 v1.0 已验证的本地 vLLM runtime；若 runtime 改变，记录新身份，但不把不同 runtime 的结果混表；
- [ ] vLLM 实际启动日志能够证明 `enable_prefix_caching=False`；
- [ ] KV scale 方案已经选定并写入唯一配置，在看到 `P_0` 质量结果后不得改变；
- [ ] 模型 revision、tokenizer revision、最大上下文、KV 预算和 generation 参数已经冻结。

若精确运行源码仍未上传，不开始 v2.0 GPU 正式实验。原因不是增加形式化门禁，而是避免再次产生“有结果、无源码”的证据断裂。

## 3. 总体资源上限

| 路径 | 正式 server 启动数 | 正式质量请求数 | 说明 |
|---|---:|---:|---|
| 无质量缺口 | 4 | 90 | `P_32/P_0` calibration 54 次 + held-out 36 次 |
| 有缺口且 `P_2` 通过 | 25 | 621 | 端点 2 + 组 8 + 单层 8 + 候选 1 + held-out 6 |
| 有缺口且到 `P_8` 才结束 | 27 | 675 | 端点 2 + 组 8 + 单层 8 + 候选 3 + held-out 6 |

上表不含一个可选的 BF16-only 难度 pilot；pilot 最多 1 次 server 启动、9 个请求。包含 pilot 时，有缺口路径的请求上限分别为 630 和 684。任何阶段都不得突破对应上限后临时追加更多层预算、更多随机对照或更长上下文。

正式数据固定为 45 个样本：calibration 27 个、held-out 18 个。每个策略只启动一次 server 并完成该 split 的全部请求；不为单个样本或单项指标重复启动。

## 4. 精简后的证据模型

v2.0 只实现以下三层证据，不再复刻 v1.0 的多重状态链：

1. `run-manifest.json`：一次记录 Git commit、源码树 hash、配置 hash、数据 hash、模型/runtime 身份和创建时间；
2. `policy-manifest.json`：每个策略一份，记录精确层集合、有效 server 参数、prefix caching 状态、请求行数、失败数和原始 JSONL hash；
3. `completed-manifest.json`：最终一次列出上述原始证据及报告，不为派生 CSV/SVG 再做独立 gate。

恢复规则只有一条：某个策略的 JSONL 可解析、行数正确、失败数为 0，且 hash 与它自己的 policy manifest 一致时才复用；否则只重跑这个策略。不存在独立 `matrix.state.json`、多层 command hash 或“为了检查检查是否成功”的额外状态文件。

## 5. 阶段 2：设计并冻结 Quality v2

### 5.1 目标

生成一份规模受控、任务有区分度、calibration/held-out 完全隔离的 45 样本数据集，并实现任务级评分和层级聚合。阶段 2 不运行 `P_0`，因此不会依据 FP8 结果反向挑数据。

### 5.2 实现任务

#### 任务 2.1：建立唯一质量配置

- [ ] 新增一个 v2.0 质量配置，集中记录模型/tokenizer、三层数据、目标长度、任务参数、seed、split、输出上限、gap 阈值和 score 版本。
- [ ] 把候选预算固定为 `{0, 2, 4, 8, 32}`。
- [ ] 明确 `P_0` 是全 FP8 端点、`P_32` 是全 BF16 端点；配置中不得再出现“固定 Auto-4 输出”。
- [ ] 配置中明确 `enable_prefix_caching=false`，并只保留一个 KV scale 字段。

建议路径：

```text
configs/v2/quality.json
```

#### 任务 2.2：生成 Easy Synthetic

- [ ] 生成 8192、16384、24576 tokens 的单针 NIAH，每个长度 2 个固定 seed，共 6 个样本。
- [ ] 每个长度 1 个样本进入 calibration、1 个进入 held-out。
- [ ] 保留 v1.0 既有答案命中语义，不在本阶段拆分或追溯修改 v1.0 EM。

#### 任务 2.3：生成 Hard Synthetic

- [ ] 实现多键/多值检索、变量追踪、聚合提取三个任务族。
- [ ] 每个任务族生成 8k、16k、24k 三种上下文，每种长度使用 seed 41、42、43，共 27 个样本。
- [ ] seed 41、42 固定进入 calibration，seed 43 固定进入 held-out。
- [ ] 用真实 tokenizer 和 chat template 控制输入长度，目标误差不超过 32 tokens，并保留 generation 空间。
- [ ] 每个样本记录任务参数和可自动评分的结构化答案。

如果预设难度没有把 BF16 保持在可用范围，允许执行一次最多 9 请求的 BF16-only pilot：每个任务/长度 1 个样本。只允许把对应任务难度调低或调高一个预注册档位，然后重新生成并永久冻结；不得查看 `P_0` 后再调。

#### 任务 2.4：冻结 Natural 子集

- [ ] 从官方 LongBench v1/LongBench-E 加载 `qasper_e` 与 `hotpotqa_e`。
- [ ] 用模型 tokenizer 过滤 chat template 后超过 24576 tokens 的样本，不对策略做差异化截断。
- [ ] 每个子集按稳定 `_id` hash 抽取 6 个样本，其中 calibration 3 个、held-out 3 个。
- [ ] 保存原始 `_id`、数据集 revision、prompt 模板和抽样规则；不得手工挑选模型容易或 FP8 容易失败的样本。

#### 任务 2.5：实现统一评分

- [ ] 为每个样本输出 `task_score ∈ [0,1]`。
- [ ] Easy 使用既有答案命中语义；Hard 使用任务匹配的准确率/集合 F1；Natural 使用对应 LongBench QA F1。
- [ ] 分别聚合 `S_easy`、`S_hard`、`S_natural`，再等权得到 `S_v2`。
- [ ] 逐样本配对差、bootstrap 95% CI 和 Answer NLL 诊断由同一汇总程序生成；NLL 不进入跨任务 `S_v2`。

#### 任务 2.6：冻结数据身份

- [ ] 生成 `calibration.jsonl` 27 行、`heldout.jsonl` 18 行和一个 `dataset-manifest.json`。
- [ ] manifest 只记录一次生成器版本、tokenizer revision、数据源 revision、配置 hash、两个 split 的行数/hash 和总数据 hash。
- [ ] 运行本地纯函数测试，证明相同配置重复生成完全相同的数据，两个 split 无重复 ID/内容。

建议路径：

```text
data/v2/quality/calibration.jsonl
data/v2/quality/heldout.jsonl
data/v2/quality/dataset-manifest.json
```

### 5.3 最小测试

只新增四组纯函数测试，不扩张 fake Docker/HTTP 测试矩阵：

1. 数据长度、规模、seed 和 split 确定性；
2. calibration/held-out 无交集；
3. 三类评分器的少量已知输入/输出；
4. 三层等权聚合与 gap 阈值边界。

### 5.4 完成判据

- [ ] 正式样本恰好 45 个，calibration 27、held-out 18；
- [ ] Hard Synthetic 恰好覆盖 3 任务 × 3 长度 × 3 seed，长度只能是 8k、16k、24k；
- [ ] Natural 恰好 12 个，两个官方子集各 6 个；
- [ ] 所有输入适配 32768 上下文且没有策略相关截断；
- [ ] 数据只产生一个最终身份 hash；
- [ ] 在产生任何 `P_0` 正式结果前，配置、数据和评分版本已提交并推送。

## 6. 阶段 3：端点质量缺口决策

### 6.1 目标

只运行 `P_32` 与 `P_0` 的 calibration 数据，自动判断是否有必要进入层搜索。首个有效请求同时承担启动 smoke，不另建 smoke 阶段。

### 6.2 实现任务

#### 任务 3.1：统一 `P_k` 配置表示

- [ ] 用一个策略结构表达 `k`、BF16 层集合和其余层的 FP8 dtype。
- [ ] `P_0` 的 BF16 层集合为空；`P_32` 的 BF16 层集合为全部 32 层。
- [ ] server 实际生效参数只写入 policy manifest 一次；不再生成多份等价 command/state 证据。
- [ ] 从启动日志读取并记录有效 KV dtype 和 `enable_prefix_caching=False`。

#### 任务 3.2：运行 calibration 端点

- [ ] 启动 `P_32`，运行全部 27 个 calibration 样本并停止 server。
- [ ] 启动 `P_0`，运行同样 27 个 calibration 样本并停止 server。
- [ ] 每个策略的第一条请求必须响应可解析、无替换字符/循环乱码，且日志证明实际 dtype；失败就停止该策略，不再额外启动两个 smoke 复现同一错误。
- [ ] 保存逐样本输出、task score、耗时和错误字段；正式结果要求失败数为 0。

#### 任务 3.3：计算缺口并做唯一决策

- [ ] 计算 `G_global`、`G_easy`、`G_hard`、`G_natural` 和逐样本配对 CI。
- [ ] 按设计文档固定阈值执行：全局不超过 0.01、Hard/Natural 各不超过 0.02、Easy 全部通过时选择 `P_0`；否则触发阶段 4。
- [ ] 生成一份 `decision.json`，写明输入 hash、四个 gap、阈值、决策和机器可读 reason code。
- [ ] 不允许人工覆盖成“为了展示 Auto 而继续搜索”。

建议结果结构：

```text
runs/<run-id>/run-manifest.json
runs/<run-id>/quality/calibration/p32.jsonl
runs/<run-id>/quality/calibration/p0.jsonl
runs/<run-id>/quality/calibration/endpoint-summary.json
runs/<run-id>/decision.json
```

### 6.3 无缺口分支

若决策为 `P_0`：

- [ ] 仅对 `P_32`、`P_0` 各运行一次 18 样本 held-out；
- [ ] 再次检查同一质量约束，不搜索层、不运行 Random-`k`；
- [ ] held-out 通过则最终输出 `P*=P_0`；
- [ ] held-out 不通过则输出 `P*=P_32`，并报告 calibration 端点决策未泛化；
- [ ] 将阶段 4 标记为 `skipped_no_quality_gap`，这代表成功停止，不代表实验失败。

### 6.4 完成判据

- [ ] 端点 calibration 各 27 行、失败 0；
- [ ] 首条请求和同一 server 的剩余请求均使用 prefix caching 关闭的实际配置；
- [ ] `decision.json` 可由原始 JSONL 和配置独立重算；
- [ ] 没有缺口时，总正式 server 启动数为 4，不产生任何选层结果；
- [ ] 有缺口时，只留下进入阶段 4 的一个明确 reason code。

## 7. 阶段 4：条件式预算与层集合搜索

### 7.1 入口条件

只有 `decision.json` 明确为 `search_required` 才执行。本阶段不得由命令行强制跳过 gap 判断进入。

### 7.2 实现任务

#### 任务 4.1：八组 coarse probe

- [ ] 固定 8 个连续层组：`[0–3]`、`[4–7]`、…、`[28–31]`。
- [ ] 每组作为一个 `P_4` 配置运行 27 个 calibration 样本，共 8 次 server 启动。
- [ ] 计算各组相对 `P_0` 的 `S_v2` 恢复量，保留前 2 组；并列按起始层号升序。

#### 任务 4.2：八层 fine probe

- [ ] 对前 2 组中的 8 层分别构造 `P_1`，每层运行 27 个 calibration 样本，共 8 次 server 启动。
- [ ] 按相对 `P_0` 的 `S_v2` 恢复量排序；并列按层号升序。
- [ ] 保存完整 8 层排名，不允许查看 held-out 后重排。

#### 任务 4.3：按预算早停

- [ ] 从同一排名构造嵌套 `P_2`、`P_4`、`P_8`。
- [ ] 先运行 `P_2` 的 27 个 calibration 样本；通过全部质量约束就停止候选搜索。
- [ ] `P_2` 不通过才运行 `P_4`；`P_4` 不通过才运行 `P_8`。
- [ ] `P_8` 仍不通过则候选输出 `P_32`；不得临时加入新预算。

#### 任务 4.4：Held-out 与随机对照

若候选为中间策略 `P_k`：

- [ ] 在 held-out 上运行 `P_32`、`P_0`、所选 `P_k` 和 3 个固定 seed 的 Random-`k`，共 6 次 server 启动、每次 18 个请求；
- [ ] 三个随机集合必须层数相同、互不相同，且在运行前由 seed 固定；
- [ ] 检查所选 `P_k` 是否再次满足质量约束；失败则最终安全输出 `P_32`，不从随机结果中另选层；
- [ ] 比较所选 `P_k` 与三个 Random-`k` 的中位数，分别报告预算选择和层排序是否得到支持。

若 `P_8` 仍失败而候选为 `P_32`：

- [ ] held-out 只运行 `P_32` 与 `P_0`，用于确认端点差距；
- [ ] 不再运行 Random-`k`，因为不存在待验证的中间层集合。

#### 任务 4.5：生成选择报告

- [ ] 输出 `selection.json`：gap、组排名、层排名、已运行预算、早停位置、候选与最终 `P*`；
- [ ] 输出质量表：三个 tier、全局分数、配对差、CI、Random-`k` 中位数；
- [ ] 输出容量表：`P_32`、`P_0`、最终 `P*` 的理论倍率和一次实测 capacity；
- [ ] 明确写出哪种结论成立：端点选择、预算选择、层选择，三者不得混为一句“Auto 优于基线”。

建议结果结构：

```text
runs/<run-id>/quality/calibration/groups/
runs/<run-id>/quality/calibration/layers/
runs/<run-id>/quality/calibration/budgets/
runs/<run-id>/quality/heldout/
runs/<run-id>/selection.json
runs/<run-id>/report/QUALITY-v2.zh-CN.md
runs/<run-id>/completed-manifest.json
```

### 7.3 完成判据

- [ ] 只有存在质量缺口时才产生 group/layer/budget 结果；
- [ ] 搜索候选严格限制为 `P_2`、`P_4`、`P_8`，且按最小 `k` 早停；
- [ ] 最多 8 个 group、8 个 single-layer、3 个 budget 和 3 个 Random-`k`；
- [ ] held-out 从未参与层排名、预算选择或阈值修改；
- [ ] 最终 `P*` 是质量约束下最小的已验证 `k`，或者安全回退 `P_32`；
- [ ] 最终报告不把 `P_0` 当成方法外的独立 FP8 方案，而是明确标为策略空间端点；
- [ ] 达到 server 启动/请求上限后必须停止并报告，不追加实验。

## 8. 失败与恢复规则

| 情况 | 处理 |
|---|---|
| 首个 `P_0` 请求再次乱码 | 保存该策略日志并停止；修复 runtime，不能用更多 smoke 掩盖 |
| 单个请求暂时性超时 | 同一策略内最多重试一次；仍失败则该策略不完整 |
| SSH 中断 | 重新运行当前策略；只有 policy manifest 与 JSONL 完整匹配时才跳过 |
| 某个策略结果损坏 | 只把该策略标记为 incomplete 并重跑，不递归重验所有已完成策略 |
| Calibration 无缺口 | 走 `P_0` held-out 分支，阶段 4 正常跳过 |
| 候选在 held-out 失败 | 最终输出 `P_32`，保留失败证据，不重新选层 |
| 达到资源上限仍无可行策略 | 停止并报告 `P_32`；不扩大搜索空间 |

## 9. 阶段 2–4 的最终交付

完成后必须能够只用以下材料回答面试追问：

1. 一份公开可取得的源码 commit；
2. 一份 45 样本数据 manifest，清楚区分 calibration 与 held-out；
3. 一份端点 gap 决策；
4. 若触发搜索，一份固定成本的组/层/预算选择轨迹；
5. 一份 held-out 质量验证和同预算随机对照；
6. 一份 `P*` 容量结果及其诚实结论。

此时再单独设计阶段 5 性能复验：只跑 `P_32`、`P_0` 与去重后的最终 `P*`，关闭 prefix caching，并在运行前冻结性能 SLO。阶段 5 不应反过来修改阶段 2–4 的质量选择。
