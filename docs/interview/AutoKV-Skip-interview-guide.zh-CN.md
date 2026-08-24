# AutoKV-Skip 面试讲解与答辩手册

更新日期：2026-08-25  
使用方法：先在服务器完成运行手册，再从 `runs/<run-id>/report/REPORT.zh-CN.md` 填入真实数值。本文不预造任何 GPU 结果。

## 1. 一页项目卡

**问题**：LLM decode 每一步都读取历史 KV Cache；长上下文和高并发下，KV 容量/带宽成为瓶颈。全 FP8 能把 KV 元素从 2 bytes 降到 1 byte，但所有层量化可能影响质量。

**方案**：在 vLLM 已有 `--kv-cache-dtype-skip-layers` 原语上，自动测量各层对 FP8 KV 量化的敏感度，选择 4 层保留 BF16，其余 28 层用 FP8 E4M3；用 NIAH、容量日志和 serving benchmark 验证。

**硬件约束**：单张 RTX A6000 48 GiB，驱动固定 535.230.02。A6000 不具备原生 FP8 Tensor Core，因此把容量和质量—容量 Pareto 作为主目标，不承诺 FP8 计算加速。

**代码贡献**：标准库 Python 控制器，负责环境 gate、镜像/模型/数据锁定、数据生成、逐层探测、选层、基线对照、隔离 benchmark、断点恢复、诊断包和报告；不 fork vLLM，不写 CUDA kernel。

**创新边界**：不是算法首创。KVTuner 已研究逐层敏感的混合精度 KV。项目的“小创新”是把当前 vLLM 的逐层跳过能力变成一套针对 A6000、低实验预算、可复现且能证伪的工程方法。

## 2. 30 秒版本

> 我做了一个叫 AutoKV-Skip 的 vLLM 推理优化小项目。全 BF16 KV Cache 质量稳但容量低，全 FP8 容量接近两倍却可能损失长上下文召回。我用 NIAH 校准集逐层做单层 BF16 恢复消融，通过召回率和答案 NLL 给层排序，自动选 4 层保留 BF16，其余 28 层用 FP8；再对比 BF16、全 FP8、5 组随机层、首层、末层和反向层。Mistral-7B 的理论 KV 容量倍率是 1.7778。整个流程锁定 vLLM 镜像 digest、模型 revision 和数据 hash，并针对不能升级的 R535 驱动做严格 smoke gate。这个工作不是新量化算法，而是一个可复现的精度—容量自动选择与系统验证闭环。

讲完停下，等面试官选择追问算法、系统或实验。

## 3. 5 分钟版本

### 第 0:00–1:00：背景与目标

LLM 推理要区分 prefill 和 decode。prefill 并行处理 prompt，往往偏计算；decode 每次一个 token，却要从显存读取所有历史 K/V，常受 KV 容量和带宽限制。FlashAttention 解决 attention 中间矩阵的 HBM I/O，PagedAttention 解决 KV 分页分配，但它们不直接减少每个 KV 元素的字节。

我选 Mistral-7B-Instruct-v0.3，它有 32 层、8 个 KV heads、head dimension 128。BF16 KV 是 131072 bytes/token；全 FP8 是 65536 bytes/token。但全层量化未必每层同样安全，所以目标是在固定 16G KV 预算下只保护少量敏感层。

### 第 1:00–2:00：算法

我把全 FP8 作为共同基线。对候选层 i，只把这一层恢复成 BF16，运行同一 NIAH 校准集：

```text
delta_recall_i = recall(single_i) - recall(all_fp8)
delta_nll_i    = nll(all_fp8) - nll(single_i)
score_i        = normalized(delta_recall_i, delta_nll_i)
```

按 score 取 top-4。quick profile 用 coarse-to-fine，只探 18 个核心配置；full 扫 32 层验证有没有漏层。这是可加性近似，不声称全局最优，因为穷举 `C(32,4)=35960` 个组合的服务启动成本不合适。

### 第 2:00–3:00：因果对照

只对比 BF16、FP8、Auto-4 不够，因为 Auto-4 多用了 BF16 字节。我增加 5 个固定种子的 Random-4、First-4、Last-4 和 Inverted-4。只有 Auto-4 优于随机中位数和反向选择，才能说明敏感度排序有效；如果所有 4 层方案都相同，只能说明混合精度有用，不能说明自动选择有用。

质量主指标是 NIAH exact recall，答案 token NLL 用于小样本破平；容量从 vLLM 启动日志解析 KV blocks/tokens；性能用固定请求分布测吞吐、TTFT 和 TPOT/ITL。所有配置固定相同的 FlashInfer backend、模型 BF16、KV 字节预算、seed 和 benchmark 参数。

### 第 3:00–4:00：系统落地

我没有改 vLLM 内核，而是使用其公开的 `--kv-cache-dtype-skip-layers`。控制器会先做 driver 精确匹配、Docker GPU 可见、容器内 SM 8.6、CLI feature、FlashInfer import、BF16/FP8 双 smoke 和重复确定性检查。

服务器驱动 535.230.02 不能更新，所以使用官方 vLLM 容器和 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`，再通过真实启动来证明兼容，不根据版本表盲猜。镜像锁 RepoDigest、模型锁 40 位 revision、数据锁 SHA-256；每阶段有状态 hash，SSH 中断后可恢复。

### 第 4:00–5:00：结果与边界

此处只读真实报告：

- BF16 / FP8 / Auto-4 实测 KV capacity：`<从 capacity.csv 读取>`；
- 三者 recall/NLL：`<从 quality.csv 读取>`；
- Auto-4 层集合：`<从 selection.json 读取>`；
- Auto-4 相对 Random-4 中位数：`<从 REPORT.zh-CN.md 读取>`；
- 吞吐、TTFT、TPOT：`<从 performance.csv 读取>`。

最后主动说明：A6000 不具备原生 FP8 Tensor Core，FP8 在这里首先是存储/带宽优化；KVTuner 是直接 prior art，所以我不宣称算法首创。我真正展示的是在硬件和实验预算约束下做出可验证优化的能力。

## 4. 15 分钟版本

### 4.1 第 0–2 分钟：先画瓶颈图

在白板画：

```text
request
   |
   +--> prefill: long Q, parallel GEMM/attention, compute + attention I/O
   |
   +--> decode: 1-token Q × all cached K/V, memory bandwidth + KV capacity

Optimization stack:
FlashAttention  -> attention I/O
PagedAttention  -> KV allocation/fragmentation
FP8 KV          -> bytes per cached token
Fusion/Graphs   -> launches/intermediate traffic
Scheduler       -> cross-request utilization
```

强调 AutoKV-Skip 只改变第三项，并固定其他项。

### 4.2 第 2–4 分钟：推导容量

写通式：

```text
B/token = 2 × L × H_kv × D_head × bytes(dtype)
```

代入 `L=32, H_kv=8, D=128`：

```text
BF16: 2 × 32 × 8 × 128 × 2 = 128 KiB/token
FP8 : 2 × 32 × 8 × 128 × 1 =  64 KiB/token
Auto: 4 layers × 4 KiB + 28 layers × 2 KiB = 72 KiB/token
128 / 72 = 1.7778
```

随后指出这只是 payload 理论，实测还含 block 对齐与 runtime overhead，所以用同一 `--kv-cache-memory-bytes 16G` 和 vLLM 日志验证。

### 4.3 第 4–7 分钟：算法与搜索预算

讲清四个选择：

1. 为什么全 FP8 是层消融基线：单层恢复直接测该层低精度造成的边际损失；
2. 为什么 recall + NLL：recall 可解释，NLL 连续，能处理小校准集并列；
3. 为什么 quick：18 次核心启动比 34 次更适合先验证机制；
4. 为什么不是穷举：35960 个组合还要跑多样本，收益不匹配面试项目预算。

也要主动暴露假设：单层边际贡献可能不满足可加性；层 A/B 单独不敏感、共同量化却可能出问题。因此选层之后必须用组合测试和随机/反向对照验证。

### 4.4 第 7–10 分钟：实验协议

展示如下证据链：

```text
doctor.json
   -> image RepoDigest + model revision lock
      -> dataset manifest/hash
         -> probe raw JSONL
            -> selection.json
               -> quality raw JSONL
                  -> perf raw JSON
                     -> Markdown/CSV/SVG report
```

每个箭头都把上游 hash 写入状态；resume 只跳过 hash 一致且输出完整的步骤。服务进程由 label 和随机端口隔离；失败时只清理自己的容器；诊断包过滤 token 且不包含模型 cache。

实验矩阵：

- probe：BF16、FP8、候选单层恢复；
- quality：BF16、FP8、Auto-4、Random-4×5、First-4、Last-4、Inverted-4；
- performance：BF16、FP8、Auto-4；
- quick：6 次 benchmark 重复；full：18 次；先 warmup。

### 4.5 第 10–12 分钟：框架与硬件决策

为什么 vLLM：当前锁定版本有逐层 skip-layers 公共接口，可以不 fork；PagedAttention 和 `vllm bench serve` 又提供容量与 serving 测量基础。

为什么固定 FlashInfer：如果 dtype 改变时 backend 自动变了，结果无法归因；同一后端能控制变量。代价是必须先实测 A6000 + FP8 + 版本组合。

为什么容器：宿主 R535 不变，容器自带配套 CUDA/PyTorch/FlashInfer，启用 forward compatibility。必须说明 compatibility 不是保证，所以有硬 gate；失败不进入正式实验。

### 4.6 第 12–14 分钟：真实结果

从报告依次展示：

1. 容量表：实测/理论倍率是否一致；
2. 层敏感度 SVG：是否集中、quick/full 是否稳定；
3. 质量表：Auto-4 是否恢复 gap；
4. random distribution：是否优于随机中位数；
5. 性能表：容量收益是否伴随 throughput/latency 变化；
6. 失败/警告：scale fallback、样本层、重复方差。

不要只报百分比。报告原始值、重复次数、聚合统计和对照条件。

### 4.7 第 14–15 分钟：贡献与下一步

贡献分三层：

- 算法工程：敏感度代理和预算选择；
- 系统工程：版本/环境锁定、隔离、恢复和诊断；
- 实验工程：对照、预注册阈值和负结果解释。

下一步只提一两个：固定真实数据 scale、0/2/4/8 层 Pareto、跨模型验证或加入层间交互的贪心前向搜索。避免许诺重写所有 kernels。

## 5. 高频追问与回答

### Q1：为什么不用 SGLang？

SGLang 很适合做 RadixAttention、prefix reuse 或 structured generation 相关项目，也支持全局 FP8 KV。这个题的增量点是“自动选择哪些层跳过量化”，当前 vLLM 锁定版本直接暴露逐层 skip-layers 参数，而当前 SGLang 公开 server 参数没有对等的逐层开关。选 vLLM 能把代码写在搜索和实验上，不必 fork 框架。结论是针对本题的集成成本，不是框架总体排名。

### Q2：为什么不是 KVTuner？

KVTuner 已经研究了 sensitivity-aware layer-wise mixed-precision KV，所以 AutoKV-Skip 不能声称发明逐层混合精度。区别在目标和约束：我只使用 vLLM 已支持的 BF16/FP8 与 skip 参数，预算固定 4 层，代理和搜索更轻；重点是 A6000/R535 下可复现部署、对照、恢复和报告。它更像“把论文方向缩成一个可审计的工程 MVP”，不是替代论文。

### Q3：这到底算不算创新？

算法新颖性弱，工程创新性适合面试项目。我可以明确列出已有原语和自己的增量：vLLM 提供量化 kernel 与逐层开关；我提供自动层排序、coarse-to-fine 预算搜索、固定字节公平比较、随机/反向因果对照、环境锁定和一键报告。若岗位要求论文级创新，这不够；若考察训推优化落地，它能展示完整方法论。

### Q4：为什么只保留 4 层？

这是预注册的简单预算点。Mistral 32 层下，4 层 BF16 + 28 层 FP8 的理论容量倍率是 1.7778×，离全 FP8 的 2×较近，又给算法足够选择空间。项目完成后可测 0/2/4/8 的 Pareto，但不能看完结果后偷偷改主预算。

### Q5：为什么用单层恢复，而不是逐层量化？

二者数学上可转换，但全 FP8 作为共同基线更贴合最终方案：我测“恢复某层能挽回多少损失”。共同基线也减少不同配置间的定义歧义。缺点是单层边际不包含组合交互，所以最终组合必须另测。

### Q6：召回率这么离散，排序靠谱吗？

召回率是可解释主指标，但小样本会并列，因此加入答案 token NLL 作为连续破平，并固定样本和 seed。仍然可能噪声大，所以用 full scan、随机对照和重复确定性检查验证。若 endpoint 无法给可信 logprob，系统应全局退化并在报告标记，而不是每个配置选择不同评分模式。

### Q7：为什么 NIAH，不用 LongBench？

NIAH 自包含、答案唯一、长度和 needle depth 可控，适合在暂时离线、实验预算有限时快速检测 KV 误差。它不是综合能力评测，所以结论只限定在该模型和该 NIAH 分布。更强结论要补 LongBench/RULER/业务集，这是外部有效性扩展。

### Q8：FP8 为什么可能不加速？

容量减半不等于端到端耗时减半。A6000 没有 Hopper 的原生 FP8 Tensor Core；attention 读取 FP8 后可能要反量化，scale 处理和 kernel 变体也有开销。只有 decode 足够 memory-bound、kernel 融合有效时，带宽节省才可能转成吞吐收益。因此我的主指标是 capacity，性能是必须测但不预设方向的次指标。

### Q9：A6000 不具备原生 FP8 Tensor Core，为什么还能存 FP8？

存储格式和矩阵乘硬件是两件事。显存可以按 8-bit 编码保存 K/V，kernel 读取后转换到支持的计算精度；这仍能降低缓存字节和读流量。没有原生 FP8 Tensor Core意味着不能把 Hopper 的 FP8 GEMM 加速数字套过来，不意味着显存不能保存 8-bit 数据。

### Q10：固定 scale 是否太粗糙？

是 MVP 的已知限制。固定 seed 的计算 scale 优先保证首次落地和可复现；双 smoke 验证其稳定性。更严谨扩展是使用代表性长上下文做离线校准、锁定 scale 文件并比较 per-tensor/per-head 粒度。但若当前 vLLM kernel 不暴露更细粒度 scale，就不能只在 Python 控制器里假装实现。

### Q11：为什么用 FlashInfer，不直接叫 FlashAttention？

服务 decode 使用 paged/ragged KV 和动态批次，和训练时规则 dense attention 的内核形状不同。FlashInfer覆盖 serving 的 prefill/decode/paged 场景；项目固定它来控制 backend。FlashAttention 的 IO-aware 思想仍是背景，但我没有修改 FlashAttention，也不会把 FlashInfer 的结果说成自己的 kernel 优化。

### Q12：PagedAttention 已经优化 KV，为什么还要量化？

PagedAttention优化逻辑 token 到物理 block 的分配、碎片和共享；KV 量化改变每个元素的字节。前者提高内存利用率，后者降低 payload，二者乘法叠加而非替代。

### Q13：容量为什么不用 `nvidia-smi` 测？

`nvidia-smi` 显示整个进程在某一时刻的显存占用，包含权重、workspace、graph 和 allocator reserve，不能直接推出可服务 token 数。vLLM 在启动时根据剩余预算和 cache block 形状计算 GPU KV blocks/tokens；在固定 16G KV 预算下，这个量更接近要比较的 capacity。

### Q14：如果模型加载后 16G 放不下怎么办？

48 GiB A6000 上 7B BF16 权重约十几 GiB，理论上有空间，但仍以 doctor/smoke 为准。命令显式固定 16G KV，避免不同 dtype 自动占用不同空闲空间。若 runtime workspace 造成 OOM，正式协议应停止并记录；若要改预算，要新建 profile 并让所有配置共同改变，不能只给某个 baseline 降预算。

### Q15：为什么随机对照要 5 个？

一个随机样本可能碰巧很好或很差。5 个固定 seed 能得到一个小分布和中位数，同时保持单卡实验预算可控。这不是统计显著性的充分样本；报告应给出全部 5 个值，不用 p-value 过度包装。

### Q16：coarse-to-fine 会不会漏掉孤立的敏感层？

会，这是 quick 的已知近似。报告必须写出 probed scope；full profile 的全 32 层扫描就是验证。若 quick/full 选层差异大，说明平滑邻域假设不成立，应把它作为方法限制，而非覆盖 quick 结果。

### Q17：为什么不做 greedy forward selection？

前向贪心每轮都在当前组合上测试候选，可以捕获一部分交互，但 4 轮最多需要约 32+31+30+29 次组合评估，质量请求与服务重启显著增多。单层 top-4 更符合小项目预算；贪心是结果显示交互明显后的合理第二版。

### Q18：如何保证 benchmark 公平？

三个性能配置由同一 builder 生成，只改变 KV dtype/skip layers；固定镜像 digest、模型 revision、FlashInfer、16G KV、请求分布、并发、seed、warmup 和重复数。每个配置独立启动 server，保存原始 vLLM bench JSON，再聚合 TTFT/TPOT/吞吐；不把启动或首次编译混入稳态。

### Q19：SSH 断了会怎样？

正式运行放在 tmux。控制器每阶段写入包含上游 hash 的状态，重跑同一 `run` 时只复用校验通过的产物；不一致则拒绝混跑。服务容器带项目/run/config label，清理只针对精确 label。

### Q20：R535 和新 CUDA 镜像真的兼容吗？

不能只凭版本号承诺。使用官方容器的 compatibility 模式是候选路径；doctor 依次验证 driver、容器 GPU、torch capability、vLLM 参数、FlashInfer import，smoke 再启动真实模型和两种 KV dtype。只有这些 gate 通过才正式实验。失败时保留诊断或按预定义条件尝试锁定 fallback 镜像，但绝不升级驱动。

## 6. 负结果怎么讲

### 6.1 没有 BF16—FP8 质量 gap

> 这是一个负结果，但不是实验失败。它说明在冻结的 Mistral/NIAH/上下文范围内，全 FP8 已经足够稳定，所以敏感层选择没有可恢复的损失。系统仍验证了接近 2× 的 KV capacity，并通过预注册的 `gap > 0.01` 条件阻止我事后夸大 Auto-4。下一步若要检验方法，应预先扩展到更长上下文或更敏感数据，而不是挑失败样本。

### 6.2 有单层信号，但 Auto-4 不胜随机

> 结果否证了“单层边际可加”这一代理。可能原因是层间交互、校准集方差或 quick 漏层。我有 Inverted-4、5 个 Random-4 和 full scan 可以区分这些解释；工程结论是当前自动选择规则不值得部署，而不是删掉对照。

### 6.3 容量上涨但延迟变慢

> FP8 在 A6000 上主要是存储压缩，不是原生 FP8 Tensor Core 计算。反量化或当前 kernel 的开销可能超过短上下文的带宽收益。只要固定预算下 KV token capacity 与理论相符，容量目标成立；性能结论应写成特定并发/长度下的 trade-off，并寻找 memory-bound 的 break-even，而不是声称全面加速。

### 6.4 运行时根本不支持该组合

> 这属于兼容性结论，不属于算法质量结论。我锁住了镜像、驱动和错误证据，能指出 failure 在 CLI feature、FlashInfer import、server startup 还是 FP8 request。因为驱动是不可变约束，我不会通过升级驱动“修好”后把结果伪装成目标环境。

## 7. 代码讲解路线

面试现场按以下顺序打开，不需要从 1000 行 CLI 顶部逐行读：

1. `configs/quick.json`：展示冻结假设、预算和实验规模；
2. `autokv/profile.py`：展示 schema/默认配置与 hash；
3. `autokv/commands.py`：展示唯一 command builder 如何只改变 KV 配置；
4. `autokv/selection.py`：展示分数、tie-break、Auto/Random/First/Last/Inverted；
5. `autokv/experiment.py`：展示 server isolation、HTTP 请求和断点恢复；
6. `autokv/benchmark.py`：展示 warmup、原始 JSON、聚合和容量解析；
7. `autokv/doctor.py`：展示不可升级驱动的 fail-closed gate；
8. `autokv/report.py`：展示 theory vs measured 与负结果判据；
9. `tests/`：展示无 GPU 的 fake Docker/HTTP 测试如何覆盖控制面；
10. `RUNBOOK.zh-CN.md`：展示未来登录服务器后无需临场猜命令。

每个文件最多讲一个设计选择，避免把“代码量”当贡献。

## 8. 可在白板现场推导的三个式子

### 8.1 KV bytes/token

```text
2 × layers × kv_heads × head_dim × dtype_bytes
```

### 8.2 混合层理论容量倍率

若 L 层中 k 层 BF16，其余 FP8：

```text
ratio_vs_bf16 = (2L) / (2k + 1(L-k)) = 2L / (L+k)
```

`L=32, k=4`：`64/36=1.7778`。

### 8.3 gap recovery

```text
recovery = (Q_auto - Q_fp8) / (Q_bf16 - Q_fp8)
```

仅当分母大于预注册阈值 0.01 时报告。若 recovery 超过 1，表示 Auto-4 在有限样本上高于 BF16，应同时报告置信范围/方差，不能解释为量化提高真实能力。

## 9. 模拟追问清单

在面试前用真实报告逐题录音回答，每题 60–90 秒：

1. FlashAttention 的 online softmax 为什么不需要物化 `N×N`？
2. FlashAttention、PagedAttention、FlashInfer 的层次分别是什么？
3. 为什么 decode 更容易 memory-bound？
4. 权重量化、激活量化和 KV 量化各影响什么？
5. FP8 E4M3 的 scale 错了会发生什么？
6. 为什么 A6000 上仍能获得 KV 容量收益？
7. 你的层分数为何可能失效？
8. random/first/last/inverted 各排除哪个解释？
9. 为什么实测容量比 1.7778 低？
10. quick 与 full 选择不同怎么办？
11. 如何证明配置之间只变了一个因素？
12. 如何确认没有跨镜像/跨模型 revision 混跑？
13. `nvidia-smi` CUDA 12.2 与容器 CUDA 有什么关系？
14. forward compatibility 为什么仍要 smoke？
15. 如果让你再写一周代码，先做 scale、交互搜索还是多模型？为什么？

## 10. 展示前检查

- 只展示 `REPORT.zh-CN.md` 中存在的真实数字；
- 能指出镜像 digest、模型 revision、dataset hash 和 run ID；
- 能口算 131072、65536、73728 bytes/token 与 1.7778；
- 能用一句话说明 A6000 的 FP8 硬件边界；
- 能承认 KVTuner prior art；
- 能解释 Auto-4 为什么必须对比相同 BF16 层数的随机/位置 baseline；
- 准备一个正结果叙事和一个负结果叙事；
- 不把运行控制器说成 CUDA kernel 开发；
- 不把 NIAH 泛化为通用模型质量；
- 不说“驱动版本看起来兼容”，而说“doctor 和 smoke 在锁定环境中实测通过/未通过”。
