# 大模型推理优化全景与 AutoKV-Skip 立项论证

更新日期：2026-08-25  
目标环境：单张 NVIDIA RTX A6000 48 GiB、驱动 535.230.02（不可更改）  
目标框架：vLLM；对照框架：SGLang

## 0. 先给结论

本项目最合适的落点不是重写 FlashAttention 内核，也不是再实现一种 4-bit 权重量化，而是利用 vLLM 已有的逐层 KV Cache 混合精度原语，完成一个小而闭环的自动决策器：

1. 用长上下文 NIAH 校准集测每一层被 FP8 量化时的质量损失；
2. 按质量代理分数选择 4 个敏感层保留 BF16 KV，其余 28 层用 FP8 E4M3；
3. 与 BF16、全 FP8、随机 4 层、前 4 层、后 4 层和反向 4 层做同预算比较；
4. 同时报告召回质量、KV token capacity、吞吐、TTFT 和 TPOT/ITL；
5. 固定镜像 digest、模型 revision、数据 hash、seed 和命令，保证结果可复现。

它的价值主要是工程研究闭环，而不是算法首创。逐层敏感度和混合精度 KV 已有明确先例，特别是 [KVTuner](https://arxiv.org/abs/2502.04420)。AutoKV-Skip 的可讲点是：在固定硬件、固定 KV 字节预算和不能升级驱动的约束下，把既有逐层原语变成自动选择、可恢复执行、有效对照和可审计报告。

选择 vLLM 而不是 SGLang 的直接原因是：当前 vLLM 文档和源码公开了 `--kv-cache-dtype-skip-layers`；当前 SGLang 的公开参数主要提供全局 `--kv-cache-dtype`，若做同一课题就要修改框架内部，明显扩大代码量。这个判断只针对本项目，不代表 vLLM 在所有服务场景都优于 SGLang。

RTX A6000 是 Ampere SM 8.6，具备 BF16/FP16/TF32/INT8 Tensor Core 路径，但 A6000 不具备原生 FP8 Tensor Core。项目中的 FP8 首先是 KV 存储压缩：减少 HBM 容量与读带宽，再在 attention 路径反量化；不能预设 FP8 算术会像 Hopper 那样直接加速。这个硬件事实决定了主成功指标应是 KV capacity 和质量—容量 Pareto，而不是“延迟一定下降”。[NVIDIA A6000 产品页](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)、[Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/11.0_GA/ampere-tuning-guide/index.html)、[Hopper 架构页](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)

## 1. 推理优化应先按阶段和瓶颈拆解

### 1.1 Prefill 与 decode 是两类问题

对自回归 Transformer，一次请求至少包含两个阶段：

- **Prefill**：并行处理整段 prompt，生成每层 KV Cache。矩阵乘规模大、并行度高，往往偏计算受限；长 prompt 的 attention 还会造成显著 HBM I/O。
- **Decode**：每步只生成一个新 token，却要读取历史所有 token 的 K/V。批量较小时算术强度低，常受显存带宽、KV 容量和调度开销限制。

因此不存在一个“统一最快”的优化。FlashAttention 主要降低 attention 中间矩阵的 HBM 往返；PagedAttention 解决 KV 内存分配和共享；KV 量化降低每 token 的缓存字节；算子融合减少中间张量和 kernel launch；continuous batching 提高跨请求利用率。面试中应先指出优化作用于哪一个阶段、哪一种资源，再谈数字。

### 1.2 单请求的 KV 容量模型

标准多头/分组查询 attention 中，每个 token 的 KV 字节近似为：

```text
bytes_per_token = 2(K,V)
                × num_layers
                × num_kv_heads
                × head_dim
                × bytes_per_element
```

Mistral-7B-Instruct-v0.3 是 32 层、8 个 KV heads、head dimension 128。忽略 block padding 和元数据时：

```text
BF16 = 2 × 32 × 8 × 128 × 2 = 131072 bytes/token
FP8  = 2 × 32 × 8 × 128 × 1 =  65536 bytes/token
```

若 4 层 BF16、28 层 FP8：

```text
Auto-4 = 2 × 8 × 128 × (4 × 2 + 28 × 1)
       = 73728 bytes/token

理论容量倍率（相对 BF16）= 131072 / 73728 = 1.7778
```

真实 vLLM 容量还受 block size、page 对齐、运行时 workspace、CUDA graph、权重和上下文上限影响，所以项目同时给出理论值和启动日志解析的实际 KV tokens；不能拿 `nvidia-smi` 峰值代替 KV capacity。

## 2. 优化技术地图

| 层次 | 代表方案 | 它优化什么 | 典型代价/风险 | 与本项目关系 |
|---|---|---|---|---|
| Attention I/O | FlashAttention 1/2/3 | 避免物化完整注意力矩阵，减少 HBM I/O | 架构、形状、mask 和数据类型约束 | 固定底座，不改 kernel |
| KV 内存管理 | PagedAttention | 分页分配、降低碎片、共享 prefix | page 元数据与调度复杂度 | vLLM 的容量基础 |
| Attention runtime | FlashInfer | prefill/decode/paged/ragged 内核与调度 | 后端支持矩阵随版本变化 | 固定为实验后端并先 smoke |
| 权重量化 | GPTQ、AWQ | 减少权重显存和读带宽 | 离线标定、kernel/模型支持 | 主实验刻意关闭，隔离变量 |
| 权重+激活量化 | SmoothQuant、FP8 W/A | 降低 GEMM 成本 | outlier、scale、硬件依赖 | 研究背景，不与 KV 混做 |
| KV Cache 量化 | vLLM FP8、KIVI、KVQuant | 降低每 token KV 字节和读带宽 | 误差、scale、反量化开销 | AutoKV-Skip 的核心原语 |
| KV Cache 压缩/淘汰 | H2O、SnapKV、StreamingLLM | 少存一部分历史 token | 可能永久丢失上下文信息 | 与“保留所有 token、降精度”严格区分 |
| 算子融合 | norm+quant、RoPE+KV update | 少 launch、少中间读写 | shape/后端专用，编译复杂 | vLLM 已实现，不重复造轮子 |
| 图与编译 | `torch.compile`、CUDA Graph | 消除 Python/launch 开销、专门化图 | 动态 shape、capture 内存 | 性能实验必须固定其策略 |
| 服务调度 | continuous batching、chunked prefill | 提高 GPU 利用率与公平性 | TTFT/TPOT/吞吐互相制约 | benchmark 固定并发/请求分布 |
| Prefix 复用 | RadixAttention、prefix cache | 避免重复 prefill | 命中率依赖工作负载 | 更适合 SGLang 的另一个项目 |
| 架构拆分 | disaggregated prefill/decode | 独立扩缩 prefill/decode | 多机传输、部署复杂 | 单 A6000 项目不做 |

## 3. FlashAttention：它快在哪里，又没有解决什么

### 3.1 核心思想

朴素 attention 会计算并常常物化 `S = QKᵀ` 和 softmax 后的 `P`。序列长度为 N 时，这些 `N×N` 中间结果在 HBM 与片上 SRAM 之间产生大量读写。FlashAttention 的关键不只是“一个更快的 CUDA kernel”，而是 IO-aware exact attention：

1. 将 Q、K、V 分块搬到片上 SRAM；
2. 用在线 softmax 保存每行的运行最大值和归一化和；
3. 遍历 K/V tiles 时更新输出，不把完整 `N×N` 矩阵写回 HBM；
4. 反向通过重计算换取更少中间存储。

结果仍是精确 attention（只受常规浮点舍入影响），不是稀疏近似。原始 [FlashAttention](https://arxiv.org/abs/2205.14135) 从 I/O 复杂度出发；[FlashAttention-2](https://arxiv.org/abs/2307.08691) 改善线程块/warp 工作划分并减少非矩阵运算；[FlashAttention-3](https://arxiv.org/abs/2407.08608) 利用 Hopper 的异步执行、TMA 和 FP8，不能把其 Hopper 数字直接外推到 A6000。

### 3.2 它与 decode、PagedAttention 的关系

FlashAttention 优化“给定 Q/K/V 如何做 attention”；PagedAttention 优化“历史 K/V 存在哪里、如何分配与共享”。二者互补。在线服务的 decode 只有很短的 query，却有很长且分页的 K/V，和训练/长 prefill 的规则 dense attention 不同，因此还需要 paged/ragged decode kernel、变长 batch 调度和 split-KV 等机制。

### 3.3 对 AutoKV-Skip 的含义

AutoKV-Skip 不声称改进 FlashAttention。项目固定 FlashInfer attention backend，是为了让 BF16、FP8 和混合配置走同一后端，将结果归因于 KV dtype/选层，而不是 backend 自动切换。启动 smoke 必须确认该版本确实接受 FP8 KV 与 skip layers；只看到 CLI 参数还不等于目标 kernel 组合能运行。

## 4. PagedAttention：把操作系统分页思想用于 KV Cache

[PagedAttention/vLLM 论文](https://arxiv.org/abs/2309.06180) 把每个序列的逻辑 KV blocks 映射到非连续物理 blocks，类似虚拟内存分页。它解决三类服务痛点：

- 不必为每个请求预留完整 `max_model_len` 连续空间；
- 减少外部碎片和过度预留；
- 可通过引用/写时复制共享 beam 或公共 prefix 的物理 blocks。

PagedAttention 本身通常不丢 token、也不改变数值精度；它是内存管理。FP8 KV 则改变每个 block 的元素类型；token eviction 则改变有哪些历史 token 被保留。三者不能混为一谈。

对实验公平性而言，所有配置固定 `--kv-cache-memory-bytes 16G`，再读取 vLLM 实际可分配的 KV blocks/tokens。这样测的是同一字节预算下不同 dtype 组合能容纳多少 token，而不是让 FP8 自动吞掉更多空闲显存后再比较。

## 5. FlashInfer：推理 attention 的可组合内核与调度层

[FlashInfer 论文](https://arxiv.org/abs/2501.01005) 面向 LLM serving 中高度多样的请求形状，覆盖 single/batched prefill、decode、paged/ragged KV、稀疏/可组合格式、负载均衡、JIT 和 CUDA Graph 集成。它并不是 FlashAttention 的简单改名；服务 decode 的分页 KV 和动态批次需要不同于规则训练张量的调度。

vLLM 提供 FlashInfer backend，公开 API 可见于 [vLLM FlashInfer backend 文档](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/attention/backends/flashinfer/)。但“框架支持 FP8”不等于每个 GPU 架构、attention 模式、prefill/decode 路径都等价支持。vLLM 自己的 FlashInfer kernel 测试也体现了版本/路径差异。因此本项目：

1. 把镜像锁到 digest，不使用 `latest`；
2. doctor 先检查参数与 import；
3. 分别启动 BF16 和 FP8 smoke；
4. 用固定请求重复两次检查确定性；
5. 只有 smoke 通过才进行 34 次层敏感度探测。

这比仅凭 README 宣称兼容更严谨。

## 6. 量化：必须先说清量化的是谁

### 6.1 权重、激活和 KV Cache 是不同对象

“模型用了 FP8/INT4”信息不足。面试时至少要回答：量化对象、位宽/格式、粒度、scale 如何得到、何时反量化、哪个 kernel 消费量化值。

- **Weight-only quantization**：权重常驻低比特，GEMM 时由专用 kernel 解码/计算。它降低模型权重占用和每 token 的权重读取；GPTQ、AWQ 属于此类。
- **Weight-activation quantization**：权重和输入激活都进入低精度矩阵乘，通常更依赖硬件和 outlier 处理。SmoothQuant 通过离线等价变换把激活难量化问题迁移到权重。
- **KV Cache quantization**：线性层和普通激活仍可用 BF16，只把 attention 历史 K/V 以低精度保存；每个 decode step 的 attention 读取低精度 KV，并由融合路径或显式步骤反量化。

AutoKV-Skip 只改变第三项。这样既能在 48 GiB 卡上运行 7B BF16 权重，又能把质量/容量变化清楚归因给 KV Cache。

### 6.2 权重量化代表工作

- [GPTQ](https://arxiv.org/abs/2210.17323) 使用近似二阶信息进行逐层权重量化，展示 3/4-bit 大模型压缩。
- [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) 观察少量显著权重通道并做 activation-aware scaling，强调部署友好的 weight-only quantization。
- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) 用数学等价的通道缩放平滑激活 outlier，使 W8A8 更容易落地。

这些方法可以成为扩展实验，但不应同时放进 AutoKV-Skip 主实验。否则权重 dtype、KV dtype、kernel 路径和显存余量同时变化，无法解释因果。

### 6.3 FP8 E4M3/E5M2 与 scale

FP8 不是简单的“浮点数砍成 8 位”。E4M3 用更多 mantissa 换较小动态范围，E5M2 用更多 exponent 换较大范围。将 BF16 K/V 映射到 FP8 时通常有：

```text
q = cast_fp8(clamp(x / scale))
x_hat = cast_compute(q) × scale
```

scale 太大浪费离散级别，太小则饱和。scale 的粒度可能是 per-tensor、per-head、per-channel、per-token，精度与元数据/kernel 复杂度不同。vLLM 的 FP8 KV 路径支持计算 scales；本项目 MVP 使用固定 seed 的 `--calculate-kv-scales`，并用重复 smoke 检查稳定性。它不是最高精度的校准方案，正式报告必须注明；外部数据校准是后续扩展，不应在看到主结果后临时改变。

### 6.4 vLLM 当前提供的逐层原语

vLLM 的 [Quantized KV Cache 文档](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) 说明 FP8 E4M3/E5M2 KV 和 scale 机制；当前主线文档进一步公开 `--kv-cache-dtype-skip-layers`，可按层索引/层类型跳过量化。[vLLM 当前文档源码](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md) 和 [FP8 KV 测试](https://github.com/vllm-project/vllm/blob/main/tests/quantization/test_fp8.py) 是比博客更可靠的实现证据。

本项目锁定 v0.26.0，并由 doctor 读取真实容器中的 CLI 帮助；这是因为主线、stable 和 tag 文档可能不同步。准确的面试表述是“我在锁定镜像中探测并使用公开 skip-layers 原语”，不是“所有 vLLM 版本都支持”。

## 7. KV Cache 量化与 KV Cache 压缩不是同一件事

### 7.1 保留全部 token、降低精度

这类方法不改变历史 token 数，只改变表示：

- [KIVI](https://arxiv.org/abs/2402.02750) 基于 K/V 不同 outlier 特性设计非对称 2-bit KV 量化；
- [KVQuant](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) 研究长上下文下的敏感度、稀疏 outlier 和逐通道等量化策略；
- [SKVQ](https://arxiv.org/abs/2405.06219) 结合通道重排、窗口和 clipping；
- [KVTuner](https://arxiv.org/abs/2502.04420) 做 sensitivity-aware、layer-wise mixed-precision KV；
- [KVmix](https://arxiv.org/abs/2506.08018) 进一步研究逐层混合精度与硬件效率。

AutoKV-Skip 属于这一类，而且与 KVTuner 最接近。它不应包装为新的混合精度理论，而是一个受约束、低代码、可复现的系统化落地。

### 7.2 丢弃/选择部分 token

另一类方案保留较少历史位置：

- [H2O](https://arxiv.org/abs/2306.14048) 基于 heavy-hitter tokens 设计动态保留策略；
- [SnapKV](https://arxiv.org/abs/2404.14469) 从 observation window 推断每个 attention head 重要的历史位置；
- [StreamingLLM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html) 保留 attention sinks 与滚动窗口以支持稳定流式生成。

这类 KV Cache 压缩可能获得比 2× 更高的有效上下文节省，但错误淘汰后信息不可恢复，任务相关性和位置分布影响大。量化的误差是数值近似，淘汰的误差是结构性缺失。两者的 baseline、质量集和失败模式不同，不应在一个“小创新”中同时做。

### 7.3 为什么本项目选量化而不选淘汰

逐层 FP8 的控制变量更干净：所有 token 都在，层数和模型结构不变，只有 KV 表示精度不同。vLLM 又已有公开逐层跳过参数，因此控制代码可保持为标准库 Python。若做 SnapKV/H2O 式策略，需要改 attention 路径、处理 paged cache 映射，并设计更丰富任务验证“丢掉的 token 不重要”，不符合少代码和一次可复现实验的目标。

## 8. 算子融合、编译与 CUDA Graph

### 8.1 算子融合的性能逻辑

两个算子单独执行，往往要把第一个结果写 HBM，再由第二个读回，还产生两次 kernel launch。融合可在寄存器/共享内存中传递中间值：

```text
unfused: read x -> op A -> write y -> read y -> op B -> write z
fused:   read x -> op A -> op B -> write z
```

decode 的矩阵规模可能较小，launch 和 memory traffic 占比更高，所以融合尤其重要。但融合不是免费午餐：寄存器压力可能降低 occupancy，动态 shape 和多种量化格式会造成 variant explosion，数值次序也可能改变。

vLLM 当前设计文档列出多类融合，例如 all-reduce+RMSNorm、attention+quant、RoPE+KV update、norm+quant 和 activation+quant，见 [vLLM Fusion Passes](https://docs.vllm.ai/en/latest/design/fusions/)。其中 attention+quant、RoPE+KV update 与 KV 路径直接相关：若量化/更新被融合，FP8 节省才不容易被额外搬运抵消。

### 8.2 `torch.compile` 与 CUDA Graph 的边界

[vLLM `torch.compile` 设计](https://docs.vllm.ai/en/stable/design/torch_compile/) 将图分割、编译后端和缓存集成到服务引擎；[vLLM CUDA Graph 设计](https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/) 则通过 capture/replay 降低 CPU launch 开销。两者和手写 kernel 融合互补：编译器擅长捕获相邻 PyTorch ops，专用 attention/quant kernel 仍由后端提供。

性能实验若一组 capture、一组 eager，就无法归因。因此 AutoKV-Skip 的所有对照由同一 command builder 生成，保持镜像、backend、graph/compile 默认和请求形状一致；报告不把框架启动/编译首轮混进稳态测量。

## 9. 服务系统层：调度通常和 kernel 一样重要

### 9.1 Continuous batching 与 chunked prefill

传统静态 batch 要等整批完成；continuous batching 在 token step 边界接纳/移除请求，减少空槽。chunked prefill 把超长 prefill 切块，与 decode 交错，从而在吞吐和 decode 延迟间调节。它们改变工作负载组合，不能只用单请求 latency 推断线上吞吐。

### 9.2 Prefix reuse 与 SGLang

[SGLang 论文](https://arxiv.org/abs/2312.07104) 的重要贡献包括 RadixAttention：以 radix tree 管理可复用 prompt/KV prefix，使结构化、多轮或共享 system prompt 的工作负载减少重复 prefill。它解决的是计算复用，不是逐层数值精度。

如果职位更关注 agent serving，一个很好的 SGLang 小项目是“基于 prefix 命中率/内存压力的 cache admission 或 eviction 策略”；但它需要真实 prefix 分布和命中率实验。AutoKV-Skip 面向模型级 KV 精度—容量权衡，vLLM 的逐层公开接口更合适。

### 9.3 Prefill/decode 解耦

prefill 偏计算、decode 偏显存带宽，集群可以把两阶段放到不同 worker。vLLM 有 [disaggregated prefill 文档](https://docs.vllm.ai/en/v0.8.0/features/disagg_prefill.html)，[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) 系统化研究了两阶段独立并行与 SLO。但单 A6000 无法给出有说服力的跨实例传输/扩缩结论，本项目不做。

### 9.4 其他成熟推理栈

- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/overview.html) 提供 NVIDIA 平台上的编译、量化、in-flight batching 和专用 kernels；
- [FasterTransformer GPT guide](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md) 展示早期成熟的 fused kernels/并行推理路径；
- [DeepSpeed Inference](https://deepspeed.readthedocs.io/en/stable/inference-init.html) 提供 kernel injection、张量并行和量化配置。

这些框架适合比较生态与部署约束，但为了一个能清楚归因的小项目，不应同时支持多个 runtime。

## 10. vLLM 与 SGLang：围绕本项目的选择，而非泛化排名

| 维度 | vLLM | SGLang | 本项目判断 |
|---|---|---|---|
| KV 管理 | PagedAttention、prefix caching、明确的 KV 内存预算 | RadixAttention、prefix reuse 是核心长项 | 两者都成熟 |
| Attention backend | FlashAttention/FlashInfer 等，版本相关 | FlashInfer/Triton/FA 等，参数丰富 | 都需 runtime smoke |
| 全局 KV dtype | 支持 FP8 等 | 支持 FP8 等 | 都能做 BF16 vs FP8 |
| 逐层 KV 跳过 | 锁定版本公开 `--kv-cache-dtype-skip-layers` | 当前公开 server 参数未见等价逐层开关 | vLLM 可零 fork 落地 |
| Benchmark | `vllm bench serve` 及 JSON 输出 | 自有 benchmark 工具 | 本项目用同容器 CLI 降低环境差异 |
| 最值得做的小创新 | 逐层精度、调度/容量策略 | prefix/radix cache 策略、structured generation | AutoKV-Skip 选 vLLM |

SGLang 的 [server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)、[quantized KV cache 文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx) 和 [attention backend 文档](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md) 显示其支持全局 KV dtype、scale 文件和多 backend。基于当前公开参数面，没有与 vLLM skip-layers 直接等价的低改动接口。若未来版本新增该参数，只需重评集成成本；这不是永久结论。

vLLM 的优势在本题中非常具体：公开参数正好构成“作用机制”，本项目只需实现“决策机制”。这符合面试项目应把代码写在增量价值上，而不是复制框架内核。

## 11. 候选小创新评估

| 候选 | 新意 | 代码量 | A6000 可验证性 | 对比清晰度 | 主要问题 |
|---|---:|---:|---:|---:|---|
| 重写 FlashAttention/Triton kernel | 中高 | 很大 | 风险高 | 可做 microbenchmark | 调试周期长，容易只做出较慢版本 |
| FlashInfer/FA backend 自动选择 | 中 | 中 | 中 | 依赖请求形状 | 框架已有 heuristic，版本耦合大 |
| 权重量化 + KV 量化联合搜索 | 中 | 大 | 中 | 变量过多 | 需要多份权重、校准和不同 kernels |
| KV token eviction | 中高 | 大 | 中 | 质量任务难设计 | 需改 attention/cache 内部 |
| FP8 scale 校准改进 | 中 | 小中 | 高 | 清晰 | 需要访问代表性校准语料，kernel scale 粒度受限 |
| 固定手工 4 层 BF16 | 低 | 极小 | 高 | 清晰 | 缺少自动决策，对照说服力弱 |
| **AutoKV-Skip** | 低中（工程） | **小** | **高** | **很清晰** | KVTuner prior art，代理可能无信号 |

推荐 AutoKV-Skip 的理由不是它最“新”，而是它有最高的完成概率和最完整的可讲闭环：硬件约束 → 数学预算 → 层敏感度测量 → 近似搜索 → 多个因果对照 → 容量/质量/性能报告 → 诚实处理负结果。

可在完成主项目后增加、但不应偷换主预注册的两个小扩展：

1. **预算曲线**：分别保留 0/2/4/8 层 BF16，画质量—容量 Pareto；只需重复 select/evaluate。
2. **真实数据 scale 校准**：把固定随机 token scale 与 NIAH/公开长文本校准 scale 比较；须另建实验版本并说明 scale API 粒度。

## 12. AutoKV-Skip 算法设计

### 12.1 问题定义

设 Transformer 有 L=32 层，`S` 是保留 BF16 KV 的层集合，预算 `|S|=4`。其余层使用 FP8 E4M3。目标可以写为：

```text
maximize    Q(S)
subject to  |S| = 4
            M_KV(S) <= fixed budget
```

其中 `Q(S)` 是 NIAH 质量，`M_KV(S)` 是配置对应的 KV 字节。直接穷举有 `C(32,4)=35,960` 种，虽然不是天文数字，但每种都要重启服务和跑长 prompt，不符合单卡小项目预算。

### 12.2 单层消融分数

以全 FP8 为共同基线。对每个候选层 `i`，只让该层回到 BF16，测质量恢复：

```text
delta_recall_i = recall({i}) - recall(∅)
delta_nll_i    = nll(∅) - nll({i})
```

召回率是主指标，但小校准集会离散且大量并列；答案 token NLL 提供连续破平信号。实现将各指标归一化后组合，并保持层号作为最终确定性 tie-break。选分数最高的 4 层：

```text
S_auto = top4_i score_i
```

这是假设单层贡献近似可加，不捕获层间相互作用。项目用 Inverted-4、random controls 和可选 full scan 揭示该假设何时失败，而不把近似包装成全局最优。

### 12.3 quick 的 coarse-to-fine 搜索

完整探测要运行：全 FP8 + BF16 + 32 个单层恢复，共 34 次启动。quick 为减少首次服务器时间：

1. coarse 阶段探测间隔层；
2. 选出高分区域；
3. 补探邻居层；
4. 去重后总计 18 个核心配置；
5. 在已探测集合中选 4 层。

full profile 扫描全部 32 层，用于验证 quick 是否漏掉敏感层。quick 的选择范围必须在报告中写成 probed subset，不能称作全 32 层最优。

### 12.4 为什么必须有六类对照

- **BF16**：质量上界、容量下界；
- **FP8**：容量上界、可能的质量下界；
- **Auto-4**：主方法；
- **Random-4 × 5**：排除“随便恢复 4 层都一样”；
- **First-4 / Last-4**：检验简单位置先验是否足够；
- **Inverted-4**：选择最不敏感的 4 层，检验排序方向是否有意义。

只有 Auto-4 优于 Random-4 中位数和 Inverted-4，才能支持“敏感度排序有用”。如果所有 4-layer 配置都相同，能支持的结论只是“混合精度本身有效”，不能支持自动选层。

## 13. 实验设计与预注册判据

### 13.1 固定项

所有正式配置必须共享：

- 同一 `vllm/vllm-openai` RepoDigest；
- 同一 Mistral 40 位 commit revision；
- 同一 NIAH dataset SHA-256；
- 模型 BF16、KV 预算 16G、FlashInfer backend；
- 同一 seed、prompt 顺序、max tokens、sampling 设置；
- 同一服务启动等待、warmup 和 benchmark 请求分布；
- 目标 GPU 唯一且驱动字符串精确为 535.230.02。

代码以 manifest、lock 和逐阶段 hash 阻止跨版本续跑。实验容器每次隔离启动，并只清理带项目 label 的自己的容器。

### 13.2 主指标与次指标

质量：

- 主指标：NIAH exact recall；
- 破平指标：答案 token NLL；若目标版本 endpoint 不提供可靠 logprob，则全局退化到确定性 echo/匹配模式，并在报告注明；
- 分层观察：不同上下文长度和 needle depth。

容量：

- 主指标：vLLM 启动日志解析的 GPU KV cache tokens/blocks；
- 理论核对：根据层 dtype 计算 bytes/token；
- 不用 `nvidia-smi` 的瞬时峰值替代容量。

性能：

- request throughput、output token throughput；
- mean/median/P99 TTFT；
- mean/median/P99 TPOT 或 ITL；
- 端到端 latency；
- quick 每配置 6 次 benchmark run，full 每配置 18 次，先 warmup。

### 13.3 预注册的解释门槛

项目不承诺正结果，先冻结以下解释规则：

1. 全 FP8 的实测 KV capacity 应接近 BF16 的 2×；考虑 block/overhead，预期下限设为 1.85×。
2. Auto-4 理论倍率是 1.7778×；实测 capacity 预期至少 1.65× BF16。
3. 只有 `Q_BF16 - Q_FP8 > 0.01` 时才讨论“回收 FP8 质量 gap”；没有 gap 时，Auto-4 没有可恢复对象。
4. 若存在 gap，Auto-4 目标是回收至少 50%，且优于 Random-4 中位数与 Inverted-4。
5. latency/throughput 是观察指标；A6000 没有原生 FP8 Tensor Core，因此不把加速作为通过条件。
6. 报告全部对照，不因结果不佳删掉某个 seed 或重新选预算。

“1.85×/1.65×”不是理论真值，而是考虑运行时对齐后的工程验收阈值；真实理论值仍分别是 2×/1.7778×。

### 13.4 NIAH 的价值与边界

NIAH 能自包含生成、答案明确、控制上下文长度和 needle 位置，适合检测长上下文 KV 误差导致的召回崩溃，也不需要下载额外 benchmark。它的弱点是任务单一、模板化、不能代表所有生成质量。

因此该项目的正确结论范围是“在冻结的 Mistral/NIAH/长度分布上”。若要发表或做生产决策，应补 LongBench/RULER/真实业务集、多个模型和更长长度；面试小项目不应把 NIAH 结果泛化成所有 LLM 质量。

## 14. A6000、R535 与 CUDA 12.2 的落地判断

`nvidia-smi` 显示的 “CUDA Version 12.2” 是驱动可支持的 CUDA API 上限之一，不表示宿主安装了完整 CUDA 12.2 toolkit，也不要求 Python 包链接宿主 `/usr/local/cuda`。正式运行使用 vLLM 官方 Docker 镜像中的 CUDA/PyTorch/FlashInfer。

vLLM 的 [GPU 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) 说明官方镜像在部分专业/数据中心 GPU 上可配合 CUDA forward compatibility，并使用 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`；NVIDIA 分别说明 [forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html) 与 [minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) 的边界。

但兼容表不能替代运行验证：PTX、kernel 架构、容器 runtime 或具体 wheel 仍可能失败。因此 runbook 将兼容性变成 gate：精确 driver check → Docker GPU 可见 → 容器 torch capability 8.6 → CLI feature probe → FlashInfer import → 两种 dtype 的真实 server smoke。任何 gate 失败都停止，不建议升级驱动。

## 15. 可能出现的负结果及其信息量

### 情形 A：BF16 与全 FP8 质量几乎相同

结论：该模型/任务/长度下 FP8 足够稳健；无法证明选层有价值。容量收益仍成立，项目可以诚实展示“自动选择没有必要”的否证结果。后续可预注册更长上下文或更敏感任务，但不能事后挑失败样本。

### 情形 B：单层分数有信号，Auto-4 却不优于随机

结论：单层可加代理不能预测 4 层组合，可能有层间交互、校准样本过小或 NLL 噪声。full scan 和随机分布能定位是 coarse search 漏层还是代理失效。

### 情形 C：容量接近理论，性能不升反降

结论：A6000 的反量化/额外 kernel 开销超过 KV 带宽节省，或当前请求未达到 memory-bound 区域。这不否定容量优化；应展示不同上下文/并发下的 break-even，而非只选最快点。

### 情形 D：FlashInfer + FP8 skip 在锁定镜像不能运行

结论：这是后端/version/hardware support gap，不是算法结果。运行手册允许且仅允许在错误分类满足条件时试 v0.19.1 fallback；若仍失败，输出诊断包并停止，不能改驱动或偷偷换 backend 后与原实验混表。

### 情形 E：随机 scale 导致重复运行不一致

结论：校准协议不可复现，层排序无效。MVP 应停止；扩展路线是离线固定 scale 文件，而不是反复重跑直到得到喜欢的排序。

## 16. 面试中最严谨的一句话边界

> AutoKV-Skip 不是算法首创；它是在 vLLM 已有逐层 FP8 KV 跳过能力上，实现的一个资源受约束自动选择与实验系统。我贡献的是可复现的敏感度代理、coarse-to-fine 搜索、等预算对照、驱动不可变条件下的部署 gate，以及能够证伪自身的报告。

这比声称“我发明了逐层混合精度 KV”更可信，也更能体现工程判断。

## 17. 一手资料索引

以下链接均用于本文具体论点；论文优先链接 arXiv/会议，框架行为优先链接官方文档/仓库。

### Attention 与服务系统

1. [FlashAttention](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2](https://arxiv.org/abs/2307.08691)
3. [FlashAttention-3](https://arxiv.org/abs/2407.08608)
4. [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
5. [FlashInfer](https://arxiv.org/abs/2501.01005)
6. [SGLang](https://arxiv.org/abs/2312.07104)
7. [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)

### 量化与 KV Cache 压缩

8. [GPTQ](https://arxiv.org/abs/2210.17323)
9. [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)
10. [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html)
11. [KIVI](https://arxiv.org/abs/2402.02750)
12. [KVQuant](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html)
13. [SKVQ](https://arxiv.org/abs/2405.06219)
14. [KVTuner](https://arxiv.org/abs/2502.04420)
15. [KVmix](https://arxiv.org/abs/2506.08018)
16. [H2O](https://arxiv.org/abs/2306.14048)
17. [SnapKV](https://arxiv.org/abs/2404.14469)
18. [StreamingLLM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html)

### vLLM、SGLang 与平台

19. [vLLM Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
20. [vLLM Quantized KV Cache 主线源码文档](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md)
21. [vLLM v0.26.0 serve CLI](https://docs.vllm.ai/en/v0.26.0/cli/serve/)
22. [vLLM bench serve](https://docs.vllm.ai/en/latest/cli/bench/serve/)
23. [vLLM Fusion Passes](https://docs.vllm.ai/en/latest/design/fusions/)
24. [vLLM `torch.compile`](https://docs.vllm.ai/en/stable/design/torch_compile/)
25. [vLLM CUDA Graphs](https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/)
26. [vLLM FlashInfer backend](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/attention/backends/flashinfer/)
27. [SGLang server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
28. [SGLang quantized KV cache](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx)
29. [SGLang attention backend](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md)
30. [vLLM GPU installation / compatibility](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
31. [vLLM troubleshooting](https://docs.vllm.ai/en/stable/usage/troubleshooting/)
32. [NVIDIA CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)
33. [NVIDIA CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
34. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
35. [NVIDIA RTX A6000](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)
36. [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/11.0_GA/ampere-tuning-guide/index.html)
37. [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)

资料截至更新日期；真正执行时以锁定镜像中 `--help`、import 和 smoke 的实测结果为准。
