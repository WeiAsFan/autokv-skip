# AutoKV-Skip：面向单卡 RTX A6000 的逐层混合精度 KV Cache 自动选择

状态：已批准并实施

日期：2026-08-24

目标框架：vLLM

目标硬件：NVIDIA RTX A6000 48 GiB（SM 8.6）

不可变宿主驱动：535.230.02

## 1. 摘要

本项目实现一个不修改 vLLM 核心源码的离线优化器：先用小规模长上下文校准集测量各 Transformer 层对 FP8 KV Cache 量化的敏感度，再在固定显存预算下，自动选择少量敏感层保留 BF16 KV Cache，其余层使用 FP8 E4M3 KV Cache。项目名为 **AutoKV-Skip**。

vLLM 已提供 `--kv-cache-dtype-skip-layers` 这一逐层混合精度原语；本项目的工作不是重复实现 FP8 内核，而是补上“哪些层应该跳过量化”的自动决策、可复现实验控制、容量/质量/性能评估以及一键报告。这使代码量保持较小，同时仍能在面试中完整讲清：问题建模、硬件约束、搜索算法、系统集成、指标设计和实验结论。

最终演示重点是一个清楚的 Pareto 对比：

- BF16 KV：质量上界，KV 容量最低；
- 全 FP8 KV：容量上界，可能有质量损失；
- AutoKV-Skip-4：仅 4 层保留 BF16，目标是在接近全 FP8 容量的同时回收一部分质量；
- 随机、首层、末层、反向选择：证明效果来自层选择，而不只是“多用了 4 层 BF16”。

## 2. 已冻结的设计决策

以下决策在实现阶段不得自行改变；若将来要改变，应作为新的实验版本，而不是覆盖本项目结果。

| 项目 | 决策 |
|---|---|
| 推理框架 | vLLM，不选 SGLang |
| vLLM 集成方式 | 调用公开 CLI/API，不维护 vLLM fork，不改 CUDA 内核 |
| 主模型 | `mistralai/Mistral-7B-Instruct-v0.3` |
| 模型计算精度 | BF16 |
| 注意力后端 | 所有实验固定 `FLASHINFER` |
| 量化对象 | 只量化 KV Cache；不量化权重和普通激活 |
| KV 量化格式 | FP8 E4M3 |
| 核心预算 | 32 层中保留 4 层 BF16 KV，其余 28 层 FP8 KV |
| 主镜像 | `vllm/vllm-openai:v0.26.0` |
| 兼容回退镜像 | `vllm/vllm-openai:v0.19.1` |
| 驱动策略 | 535.230.02 原样保留；禁止升级、降级或重装 |
| CUDA 策略 | 只使用容器内 CUDA 与官方 forward-compat 库；不依赖宿主 CUDA 12.2 toolkit |
| Scale 策略 | MVP 使用 `--calculate-kv-scales` 与固定 seed 的一次性随机 token 标定 |
| 质量任务 | 自包含的 Needle-in-a-Haystack（NIAH）长上下文召回集 |
| 主质量指标 | 精确召回率；答案 token NLL 作为连续型排序/破平指标 |
| 主容量指标 | vLLM 启动日志中的可用 KV blocks/tokens；不使用 `nvidia-smi` 峰值代替容量 |
| 主性能指标 | 吞吐、TTFT、ITL/TPOT、端到端延迟 |
| 默认运行方式 | 单卡、并发 1 做质量；固定 KV 字节预算做性能 |
| 随机种子 | 42；随机对照额外使用固定种子表 `[11, 23, 37, 53, 71]` |

选择 vLLM 而不是 SGLang 的原因是：vLLM 当前公开提供逐层 KV 量化跳过参数，能直接承载本项目最小创新；SGLang 更适合做 attention backend 调度或 radix cache 方向的小项目，但会增加本项目的框架改动量，偏离“少代码、简单实验、清楚对比”的目标。

## 3. 目标与非目标

### 3.1 必须达到的目标

1. 在不能修改驱动的 RTX A6000 上，给出可执行、可恢复、可诊断的完整流程。
2. 自动寻找 BF16 KV 层集合，而不是手工挑层。
3. 使用相同模型、后端、seed、提示和 KV 显存预算做公平比较。
4. 输出机器可读原始记录和面试可直接使用的中文 Markdown 报告。
5. 所有宿主端控制代码使用 Python 标准库；GPU 依赖封装在固定 Docker 镜像中。
6. 支持中断后继续，已经完成且校验通过的配置不会重复跑。
7. 提供 dry-run，使用户在真正占用 GPU 前能看到完整命令和预计配置数。

### 3.2 明确不做的事情

- 不更新 NVIDIA 驱动，不安装新的宿主 CUDA toolkit；
- 不实现 FlashAttention/FlashInfer CUDA kernel；
- 不实现 FP8 数据类型或量化算子；
- 不量化模型权重，不把 AWQ/GPTQ/INT4 引入主实验；
- 不做训练或微调；
- 不做多 GPU、张量并行或分布式调度；
- 不声称 A6000 具有原生 FP8 Tensor Core 计算加速；
- 不以单次延迟变快作为项目成功的必要条件；
- 不在主流程中自动安装 Docker、NVIDIA Container Toolkit 或系统包；
- 不把镜像 `latest`、未固定 Git main 或可变数据集快照用于正式结果。

## 4. 硬件与驱动约束

### 4.1 不可变约束

服务器已知配置：

- GPU：NVIDIA RTX A6000，48 GiB；
- 架构：Ampere，compute capability 8.6；
- 驱动：535.230.02，不能更新；
- `nvidia-smi` 所示 CUDA compatibility：12.2。

项目中的任何脚本都不得执行或建议自动执行以下操作：

- 安装、升级、降级或删除 `nvidia-driver*`；
- 安装 CUDA runfile；
- 修改内核模块、DKMS 或 Secure Boot；
- 把宿主 `/usr/local/cuda` 挂进 vLLM 容器；
- 用“升级驱动”处理 CUDA/PTX 报错。

### 4.2 兼容方案

vLLM 官方 Docker 镜像包含 CUDA forward compatibility 库。所有 GPU 容器都必须设置：

```text
VLLM_ENABLE_CUDA_COMPATIBILITY=1
```

R535 是官方文档列出的兼容宿主驱动系列之一，RTX A6000 属于受支持的专业卡类别。容器内 CUDA/PyTorch/FlashInfer 由镜像自身提供，宿主 CUDA 12.2 不参与 Python/CUDA 依赖解析。

兼容性必须由实际 preflight 证明，不能只凭版本号推断。preflight 至少验证：

1. 宿主驱动字符串精确等于 `535.230.02`；
2. `docker info` 可用；
3. NVIDIA container runtime 能看到唯一一张 A6000；
4. 容器内 `torch.cuda.is_available()` 为真；
5. 容器识别 compute capability 为 `(8, 6)`；
6. vLLM CLI 同时包含 `--attention-backend`、`--kv-cache-dtype`、`--kv-cache-dtype-skip-layers`、`--kv-cache-memory-bytes` 和 `--calculate-kv-scales`；
7. FlashInfer 可以导入；
8. 使用小长度启动 Mistral smoke server 后，日志明确显示 `FLASHINFER` 和 FP8 KV Cache。
9. 用相同 seed 独立启动两次 FP8 smoke，固定请求的输出、token 数和解析出的 KV 容量一致；若不一致，不进入层排序。

## 5. 镜像选择与锁定协议

镜像决策必须是确定性的：

1. 先拉取并探测 `vllm/vllm-openai:v0.26.0`；
2. 只有当失败被分类为 CUDA 初始化、PTX toolchain、缺少要求的 CLI 参数或 FlashInfer 导入失败时，才探测 `vllm/vllm-openai:v0.19.1`；
3. 网络下载失败、Hugging Face 认证失败、磁盘不足或模型文件损坏不触发镜像回退，因为这些问题与 vLLM 版本无关；
4. 第一个通过全部 gate 的镜像被写入 `runs/_environment/lock.json`，同时记录 tag、image ID、RepoDigest、容器内 vLLM/PyTorch/CUDA/FlashInfer 版本；
5. 后续任何实验只能使用 lock 中的 image ID/RepoDigest，不能重新解析 tag；
6. 一次正式运行内绝不混用两个 vLLM 版本；
7. 若两个镜像都失败，脚本停止并生成诊断包；不编译源码，不修改驱动。

正式报告必须显示最终镜像 digest，而不仅是 tag。
主镜像与回退镜像的结果也不得横向拼成一张性能表；若日后用另一版本复现实验，必须创建新 run 并明确标注“跨版本观察”。

## 6. 为什么 A6000 仍可研究 FP8 KV Cache

A6000 没有 Hopper/Ada 的原生 FP8 Tensor Core 路径，因此本项目不预设 FP8 会降低所有延迟。Ampere 上的价值主要来自：

- KV Cache 每个元素由 2 bytes 降到 1 byte；
- 相同显存可容纳更长上下文或更高并发；
- decode 阶段读取 KV Cache 的显存流量下降；
- FlashInfer 可在 Ampere 上把存储的 FP8 KV 转换为 BF16/FP16 后计算。

转换开销可能抵消一部分带宽收益，甚至令某些小 batch 延迟变慢。这不是实验失败，而是应在报告中解释的硬件结论。项目的主收益指标因此是容量/质量 Pareto，性能是观察指标。

## 7. KV Cache 显存模型

对标准 GQA decoder，每 token、每层 KV Cache 字节数为：

```text
bytes_per_token_per_layer
  = 2(K 和 V) × num_kv_heads × head_dim × dtype_bytes
```

Mistral-7B-Instruct-v0.3 使用 32 层、8 个 KV heads、head dimension 128：

```text
BF16 每层每 token = 2 × 8 × 128 × 2 = 4096 bytes
FP8  每层每 token = 2 × 8 × 128 × 1 = 2048 bytes

全 BF16 = 32 × 4096 = 131072 bytes/token = 128 KiB/token
全 FP8  = 32 × 2048 =  65536 bytes/token =  64 KiB/token
```

若 `k` 层保留 BF16、其余层 FP8：

```text
mixed_bytes(k) = k × 4096 + (32-k) × 2048
ratio_to_bf16(k) = (32+k) / 64
ideal_capacity_gain(k) = 64 / (32+k)
```

核心配置 `k=4`：

```text
mixed_bytes(4) = 73728 bytes/token = 72 KiB/token
相对 BF16 占用 = 56.25%
理论容量提升 = 1.7778×
```

vLLM 的 block 对齐、cache group 统一和运行时保留空间会造成实测偏差，因此报告同时显示理论值和启动日志解析出的实测值。禁止用 `nvidia-smi` 总占用直接推导 KV 容量，因为 vLLM 会主动填满分配给 KV Cache 的预算。

## 8. AutoKV-Skip 算法

### 8.1 问题定义

令模型共有 `L=32` 层，`S` 是保留 BF16 KV 的层集合，`|S|=k=4`。其余层采用 FP8。目标是在固定 BF16 层预算下最大化长上下文质量：

```text
maximize Q(S)
subject to |S| = 4
```

完整组合数 `C(32,4)=35960`，逐一实验没有必要。项目采用离线敏感度代理，把每层的边际质量贡献近似为可加，然后用排序选择。

### 8.2 单层敏感度

先运行全 FP8 配置，得到质量 `Q_fp8`。对候选层 `i`，只让第 `i` 层回到 BF16，其余 31 层仍为 FP8，得到 `Q_i`：

```text
sensitivity_i = Q_i - Q_fp8
```

Mistral 每层 KV shape 相同，单层 BF16 的额外字节相同，因此无需再除以 cost；代码仍保留一般化的 `gain_per_extra_byte` 字段，便于面试时说明如何推广到非均匀层模型。

选择敏感度最大的 4 层形成 `S_auto`，然后必须实际启动联合配置并测 `Q(S_auto)`。单层得分只用于提出候选，最终结论只根据联合配置实测值。

### 8.3 两级搜索（默认 quick profile）

为了减少 A6000 上反复加载模型的时间，默认采用 coarse-to-fine 搜索：

1. 把 32 层按顺序分成 8 组，每组 4 层；
2. 对每组运行“该组 4 层 BF16、其余 FP8”，得到组敏感度；
3. 选组得分最高的 2 组；
4. 仅对这 2 组中的 8 个层做单层探测；
5. 从 8 个层中选出 top-4；
6. 实测联合 top-4 配置。

这需要 `1 + 8 + 8 + 1 = 18` 个核心配置（全 FP8、8 个组、8 个层、联合配置），而完整扫描需要 `1 + 32 + 1 = 34` 个核心配置。两级搜索少约 47% 的模型启动次数。

完整扫描作为 `full` profile：探测 32 个层，用于验证 coarse-to-fine 是否漏掉跨组敏感层。默认项目展示 quick 结果；面试前若有一晚 GPU 时间，再运行 full。

### 8.4 质量函数

每个 NIAH 样本要求模型输出一个多 token 验证码五次，以保证 decode 阶段确实读取长 KV Cache。每个配置计算：

- `EM`：标准化后是否包含正确的五次验证码，取样本均值；
- `answer_nll`：teacher-forced 正确答案 token 的平均负对数似然；
- `P = exp(-min(answer_nll, 20))`：把 NLL 映射到 `(0,1]`；
- `Q = 0.8 × EM + 0.2 × P`。

EM 是面试报告中的主质量指标，`Q` 仅用于层排序和 EM 相同时破平。若某个 vLLM 版本的兼容 OpenAI endpoint 无法可靠返回 prompt logprobs，则自动退化为：

```text
Q = EM - 0.001 × normalized_edit_distance
```

退化会记录在 manifest 和报告中，且所有配置必须使用同一质量函数。

### 8.5 对照组

最终实验至少包含：

| 配置 | BF16 KV 层 | 目的 |
|---|---:|---|
| BF16 | 32 | 质量上界、容量下界 |
| FP8 | 0 | 容量上界、量化质量基线 |
| Auto-4 | 4 | 本项目方案 |
| Random-4 × 5 | 每次 4 | 同成本随机基线 |
| First-4 | 0–3 | 常见启发式基线 |
| Last-4 | 28–31 | 常见启发式基线 |
| Inverted-4 | 已探测候选中单层得分最低 4 层 | 验证敏感度排序方向 |

所有混合配置严格具有相同的 4 层 BF16 成本。
五个 Random-4 集合由固定 seed 表生成，必须彼此不同；若某个集合恰好等于 Auto-4、First-4 或 Last-4，则按确定性规则继续取下一个 seed，避免重复对照。quick profile 的 Inverted-4 只在被细查的 8 个候选层中定义；full profile 的 Inverted-4 在全部 32 层中定义，报告必须注明这一差别。

## 9. 系统架构

项目分为宿主控制平面和容器数据平面：

```text
用户命令
  -> Python 标准库 CLI
      -> 环境检查与镜像锁定
      -> 生成实验矩阵
      -> 启停固定版本 vLLM Docker server
      -> HTTP 发送 NIAH/benchmark 请求
      -> 原子写入 JSONL/manifest
      -> 选择 BF16 层集合
      -> 统计与生成 Markdown/SVG 报告
  -> Docker 内 vLLM + PyTorch + CUDA compatibility + FlashInfer
      -> RTX A6000
```

控制器不导入 vLLM、torch 或 transformers。它只依赖 Python 3 标准库、Docker CLI 和 HTTP API，因此宿主环境简单，也不会与容器 CUDA 发生冲突。

## 10. 组件与职责

实现阶段使用一个入口：

```text
python3 -m autokv <command>
```

组件职责如下：

| 命令 | 职责 | 主要输出 |
|---|---|---|
| `doctor` | 只读检查硬件、驱动、Docker、镜像和 vLLM feature gate | `runs/_environment/doctor.json`、诊断日志 |
| `lock-image` | 按协议选择镜像并锁 digest | `runs/_environment/lock.json` |
| `make-data` | 生成固定 NIAH 样本并记录 hash | `data/niah/*.jsonl`、manifest |
| `dry-run` | 展示配置、容器命令、预计启动次数和磁盘路径 | 终端输出，不运行 GPU 实验 |
| `probe` | 运行 group/layer 敏感度实验 | `runs/<id>/probe/*.jsonl` |
| `select` | 计算排序和 top-k，生成对照层集合 | `runs/<id>/selection.json` |
| `evaluate` | 跑 BF16/FP8/Auto/controls 的最终质量集 | `runs/<id>/quality/*.jsonl` |
| `benchmark` | 固定 KV 字节预算运行容量和服务性能测试 | `runs/<id>/perf/*` |
| `report` | 统计 bootstrap CI，生成表格和 SVG | `runs/<id>/report/REPORT.zh-CN.md` |
| `run` | 按状态机串联上述命令，可断点续跑 | 全部输出 |
| `diagnose` | 收集失败上下文，不修改系统 | `runs/<id>/diagnostics/*.tar.gz` |

`run --profile quick` 是默认的一键路径；每个子命令仍可独立运行，便于用户理解和面试演示。

## 11. 数据流与状态机

### 11.1 正常路径

1. `doctor` 记录宿主事实，不做安装；
2. `lock-image` 选择并钉死一个镜像；
3. `make-data` 创建确定性校准集、最终质量集和性能请求规格；
4. `probe` 先跑 FP8，再跑组探测和层探测；
5. `select` 产生 Auto-4 和固定对照层集合；
6. `evaluate` 对所有配置跑相同的最终样本；
7. `benchmark` 对 BF16、FP8、Auto-4 跑固定工作负载；
8. `report` 汇总理论容量、实测容量、质量和性能；
9. `run` 写入完成标记及所有文件 hash。

### 11.2 状态与恢复

每个配置拥有唯一 canonical ID，由以下字段 hash 得到：

```text
image_digest + model_revision + backend + dtype + skip_layers
+ max_model_len + kv_cache_memory_bytes + seed + dataset_hash
```

每个配置只在以下条件全部成立时视为完成：

- 结果 JSONL 可解析；
- 请求数与 manifest 相符；
- server 日志通过 backend/dtype/skip-layer 断言；
- 结果文件 SHA-256 与状态文件一致；
- 容器已正常退出。

写文件采用“临时文件 + fsync + rename”原子提交。中断后重新执行同一命令会跳过已完成配置，只重跑不完整配置。`--force` 只能重跑精确配置 ID，不能删除整个 runs 目录。

## 12. vLLM server 固定参数

正式实验的共同参数为：

```text
--model mistralai/Mistral-7B-Instruct-v0.3
--dtype bfloat16
--attention-backend FLASHINFER
--max-model-len 32768
--kv-cache-memory-bytes 16G
--seed 42
--calculate-kv-scales
--tensor-parallel-size 1
```

每个配置只改变：

- BF16：`--kv-cache-dtype bfloat16`，无 skip layers；
- FP8：`--kv-cache-dtype fp8_e4m3`，无 skip layers；
- 混合：`--kv-cache-dtype fp8_e4m3 --kv-cache-dtype-skip-layers <indices...>`。

BF16 配置即使 CLI 接受 `--calculate-kv-scales` 也不需要 scales；命令生成器应只在 FP8/混合配置附加该 flag，避免无意义警告。

每次 server 启动后，控制器必须从日志验证实际 backend 和 KV dtype，不能只相信输入参数。质量请求固定 `temperature=0`、`top_p=1`、并发 1、`max_tokens=24`。不同配置之间必须完全重启 engine，因为 KV Cache layout 在初始化时确定。

## 13. NIAH 数据设计

### 13.1 样本结构

每个样本由以下部分组成：

1. 明确任务说明；
2. 大量确定性 filler 文本；
3. 在指定深度插入唯一 needle，例如 `VERIFICATION-CODE: ZEBRA-4821`；
4. 末尾询问验证码，并要求输出五次、以 `|` 分隔；
5. 期望答案，例如 `ZEBRA-4821|ZEBRA-4821|...`。

验证码、filler 顺序、上下文长度和插入位置由 seed 决定。生成后存 JSONL 和 SHA-256，正式运行绝不现场随机生成。

### 13.2 quick profile

- 层探测集：8K、16K tokens；needle 深度 20%、50%、80%；每点 1 个 seed，共 6 个样本；
- 最终质量集：8K、16K、30K；深度 10%、50%、90%；每点 2 个 seed，共 18 个样本；
- 性能集：1K、8K、16K 输入长度；固定短输出和长输出两种场景。

### 13.3 full profile

- 探测所有 32 层；
- 最终质量额外加入 4K 和 24K；
- 每个长度/深度使用 3 个 seed；
- 性能每个场景至少 3 个独立重复。

准确 token 长度通过正在运行的同一 vLLM tokenizer endpoint 测量，并用二分调整 filler 数量。允许误差不超过目标长度的 0.5%，且总 token 数必须给 24 个输出 tokens 和系统模板留出余量。

## 14. 性能与容量实验

### 14.1 固定显存预算

所有配置统一使用 `--kv-cache-memory-bytes 16G`。这样模型权重、后端和 KV 物理预算不变，能够比较不同 dtype 下可容纳的 tokens/blocks。

容量从 vLLM 初始化日志解析，并同时用理论公式交叉校验：

- 全 FP8 理论上约为 BF16 的 2×；
- Auto-4 理论上约为 BF16 的 1.7778×。

若实测偏差超过 10%，报告必须显示 cache block 对齐或统一 page size 的日志证据，不可静默忽略。

### 14.2 服务性能

优先调用镜像内同版本 `vllm bench serve`，避免 host 安装另一个 vLLM。每个配置：

1. server ready 后执行固定 warmup；
2. warmup 数据不计入结果；
3. quick 运行 1 次正式批次，full 运行 3 次；
4. 记录 request throughput、output token throughput、median/p90/p99 TTFT、ITL/TPOT 和 E2E latency；
5. 同时采样 `nvidia-smi dmon` 的功耗、显存和利用率，仅作解释变量。

性能主对比只包含 BF16、FP8、Auto-4，以控制总时长。质量对照组不需要全部跑性能。

## 15. 统计与结论规则

对同一批 prompt 使用配对比较。报告使用固定 seed 的 bootstrap（10,000 次）计算 95% 置信区间。

质量 gap recovery 定义为：

```text
recovery = (Q_auto - Q_fp8) / (Q_bf16 - Q_fp8)
```

仅当 `Q_bf16 - Q_fp8 > 0.01` 时计算 recovery。若分母过小，报告必须写“在本任务上未观察到足够的 FP8 质量缺口”，不能用不稳定比值夸大效果。

预先注册的成功标准：

1. FP8 实测 KV token capacity 至少达到 BF16 的 1.85×；
2. Auto-4 实测 capacity 至少达到 BF16 的 1.65×；
3. 若存在可测质量缺口，Auto-4 回收至少 50% 的 `Q` gap；
4. Auto-4 的 `Q` 高于 5 个 Random-4 的中位数和 Inverted-4；
5. 所有结论对应同一锁定镜像和同一 dataset hash；
6. 报告不把不显著的差异描述为提升。

若第 3、4 条不成立，项目仍输出完整负结果。面试表述应是“层敏感度代理在该模型/任务上未优于随机”，随后分析非加性层交互、标定集规模和随机 scale 的限制，而不能更换指标后重新包装成功。

## 16. Scale 标定范围

MVP 固定使用 vLLM 官方支持的 random-token on-the-fly scale：

```text
--calculate-kv-scales --seed 42
```

原因：

- 不创建第二份 14+ GiB checkpoint；
- 不引入 LLM Compressor、datasets、transformers 版本耦合；
- 保持“少代码、简单操作”的项目边界；
- 所有实验使用同一机制和 seed，适合比较层选择。

限制必须写进报告：scale 只从 warmup 的随机 token batch 估计，不等同于真实数据集标定，绝对精度不应代表 FP8 KV 的最佳上限。

在正式 probe 前，控制器必须执行两次独立、同 seed 的 FP8 smoke。若确定性输出或容量不一致，MVP 立即停止并给出数据集标定增强版的精确入口；不得在不稳定 scale 上继续排序并挑选“最好看”的一次结果。

数据集标定被定义为后续增强而非完成 MVP 的条件。增强版可用 LLM Compressor 的 KV-only `kv_cache_scheme` 在固定 Ultrachat 子集上生成 per-tensor scales，但不得同时量化权重或普通激活，也不得与 MVP 结果混在同一 run ID 中。

## 17. 结果目录与数据契约

建议项目结构：

```text
autokv-skip/
  README.md
  RUNBOOK.zh-CN.md
  pyproject.toml
  autokv/
    cli.py
    config.py
    doctor.py
    docker.py
    niah.py
    probe.py
    select.py
    evaluate.py
    benchmark.py
    report.py
  tests/
  configs/
    quick.json
    full.json
  data/
    niah/
  runs/
  docs/
    research/
    superpowers/specs/
```

每条原始请求结果至少包含：

```text
schema_version, run_id, config_id, image_digest, model_revision,
backend, kv_dtype, skip_layers, seed, sample_id, prompt_tokens,
output_tokens, expected, output_text, exact_match, edit_distance,
answer_nll, ttft_ms, e2e_ms, timestamp_utc
```

probe/final 质量请求采用非流式 completions，因此这些 JSONL 行的
`ttft_ms` 必须显式写为 `null`，`e2e_ms` 才是该请求的可用延迟值；不得把
E2E 伪装成 TTFT。TTFT/TPOT/ITL 只从 `vllm bench serve` 的性能原始 JSON
和逐场景 CSV 报告。

密钥不得写入结果。只记录 `HF_TOKEN` 是否存在，不记录值。

## 18. 错误处理决策表

| 故障 | 自动行为 | 用户下一步 |
|---|---|---|
| 驱动不是 535.230.02 | 立即停止，生成 doctor 报告 | 核对是否连接了目标服务器；不改驱动 |
| Docker 不存在/daemon 未启动 | 停止，不自动安装 | 按运行手册的宿主前置章节配置 Docker |
| NVIDIA runtime 看不到 GPU | 停止并收集 `docker info`、`nvidia-smi` | 按运行手册检查 NVIDIA Container Toolkit；不改驱动 |
| 主镜像 CUDA/PTX 不兼容 | 仅探测回退镜像 | 若回退通过则锁回退镜像 |
| 两个镜像都 CUDA/PTX 失败 | 停止并打包诊断 | 不编译、不升级驱动；本次环境不满足 gate |
| 镜像拉取失败 | 指数退避重试 3 次后停止 | 检查网络/registry；不切版本掩盖网络问题 |
| HF 401/403 | 停止，日志隐藏 token | 重新设置 `HF_TOKEN` 后执行同一命令 |
| 磁盘不足 | 在下载前停止 | 释放项目外磁盘或改显式 cache 路径 |
| server OOM | 保存日志；只允许降低并发/benchmark 请求数 | 不改变 16G KV 预算或模型来伪造对比 |
| 某配置请求超时 | 重启精确配置，最多 2 次 | 仍失败则标记失败并生成诊断 |
| NLL endpoint 不兼容 | 全实验统一退化到 EM+编辑距离 | 报告注明退化，不混用指标 |
| 中途断电/SSH 断开 | 原子状态保留 | 重跑相同 `run` 命令自动续跑 |
| Auto-4 无收益 | 保留负结果 | 运行 full profile 或作为局限讨论，不改成功标准 |

## 19. 安全与可重复性

1. 所有容器名以 `autokv-` 开头，脚本只管理自己创建且 label 匹配的容器；
2. 不执行宽泛的 `docker system prune`；
3. 不删除 Hugging Face cache；
4. 所有路径先解析为项目根目录下的绝对路径；
5. `diagnose` 默认做复制和压缩，不移动原始结果；
6. 命令日志中的 token、Authorization header 和 URL query secret 必须脱敏；
7. 正式 manifest 记录 Git commit、配置文件 hash、镜像 digest、模型 revision、数据 hash 和时区；
8. 模型 revision 在第一次解析后锁为 commit hash；
9. 正式数据与 quick/full 配置文件一旦进入 run，不允许原地编辑；新参数产生新 run ID；
10. 所有时间使用 UTC 存储，报告可显示 Asia/Shanghai。

## 20. 测试策略

由于当前无法访问 A6000，交付前的验证分为两层。

### 20.1 本地可完成的测试

- 配置解析和 schema 校验；
- Mistral KV 字节公式；
- skip-layer 参数生成和排序；
- coarse-to-fine 候选集合；
- 随机基线可重复性；
- NIAH 生成、needle 位置和答案标准化；
- OpenAI response 解析；
- NLL fallback；
- bootstrap 置信区间；
- Docker 命令转义、secret 脱敏和 dry-run snapshot；
- 原子状态写入与中断恢复；
- 伪造日志中的 backend/dtype/capacity 解析；
- 报告生成 golden test；
- 禁止驱动修改命令的静态检查。

### 20.2 服务器上必须完成的集成 gate

- 真正的 R535 CUDA compatibility 初始化；
- FlashInfer + FP8 E4M3 smoke；
- 单层 skip 日志与实际 KV dtype；
- 8K NIAH 单样本；
- BF16/FP8/Auto-4 的 capacity 日志；
- quick profile 全流程；
- 报告中所有配置的镜像 digest 一致。

本地测试通过不等于 GPU 集成通过。最终 README 和报告会明确区分“代码已验证”和“尚待目标服务器验证”。

## 21. 预期操作顺序

实现后的运行手册必须让用户按以下顺序执行，且每一步都给出：精确命令、预计时长、预期输出、成功判据和失败分支。

1. 把项目复制到 Linux 服务器；
2. 只读检查 GPU、驱动、内存、磁盘、Docker 和 NVIDIA runtime；
3. 设置项目目录及可选 `HF_TOKEN`；
4. 运行 `doctor`；
5. 锁定并记录可用镜像；
6. 下载/缓存固定模型 revision；
7. 生成并检查 NIAH 数据 hash；
8. 先执行 dry-run；
9. 运行 10 分钟级 smoke；
10. 用 `tmux` 或 `systemd-run --user` 启动 quick profile；
11. 查看进度或在 SSH 断开后恢复；
12. 生成报告并检查 acceptance gate；
13. 若 quick 结果有信号，选择性运行 full profile；
14. 导出报告、CSV、SVG、环境 lock 和诊断信息用于面试材料。

主流程必须设计成用户以后只需要复制一组按顺序编号的命令，不需要临场选择参数。

## 22. 预计资源与时长

在 A6000 上的保守估算（实际取决于网络、CPU、磁盘和 vLLM 版本）：

- 镜像 + 模型首次下载：20–90 分钟；
- preflight + smoke：10–30 分钟；
- quick 层搜索：约 2–5 小时；
- quick 最终质量与性能：约 3–7 小时；
- quick 总 GPU 时间：约 5–12 小时；
- full profile：约 12–24 小时。

建议宿主至少有 64 GiB RAM、80 GiB 可用磁盘、Linux x86_64、Docker Engine 与已配置的 NVIDIA Container Toolkit。资源不足由 doctor 在下载/运行前报告。

## 23. 面试讲解主线

项目完成后应能用五分钟讲清：

1. **瓶颈**：长上下文 decode 的 KV Cache 容量和显存带宽；
2. **已有原语**：FlashInfer、PagedAttention、FP8 KV 和逐层 skip；
3. **观察**：不同层对 KV 量化误差的敏感度不同；
4. **方法**：小校准集估计边际敏感度，预算约束下选 top-k；
5. **工程**：不 fork vLLM，用公开参数做稳定集成，固定 backend 和镜像 digest；
6. **公平性**：相同 BF16 层数的随机/首层/末层/反向对照；
7. **结果**：展示质量—容量—延迟三维 Pareto；
8. **硬件理解**：A6000 的 FP8 是存储压缩路径，不夸大为原生 FP8 计算；
9. **局限**：单层可加代理、随机 token scale、NIAH 外部效度；
10. **下一步**：dataset-calibrated scales、非均匀 layer cost、联合搜索或在线 workload-aware policy。

## 24. 验收条件

实现只有同时满足以下条件才算完成：

- 存在中文逐命令运行手册，任何主步骤都没有未完成占位符；
- 所有正式镜像、模型和数据可锁定到不可变标识；
- 项目没有修改驱动或宿主 CUDA 的代码路径；
- 单元测试覆盖核心算法、状态恢复和命令安全；
- dry-run 能输出 quick/full 的精确配置数；
- server 日志断言能阻止 backend 或 dtype 静默回退；
- quick 流程可断点续跑并生成完整报告；
- 报告区分理论容量、实测容量、质量和性能；
- 报告对无显著提升或负结果使用诚实表述；
- 用户只需按运行手册操作，不需要回到本对话补问缺失参数。

## 25. 主要资料

- vLLM GPU 安装与旧驱动 CUDA compatibility：<https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
- vLLM 旧驱动/PTX 故障排查：<https://docs.vllm.ai/en/latest/usage/troubleshooting/>
- vLLM v0.26.0 `serve` 参数：<https://docs.vllm.ai/en/v0.26.0/cli/serve/>
- vLLM quantized KV Cache 文档源码：<https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md>
- vLLM FlashInfer backend 支持的 KV dtype 与 SM 范围：<https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/attention/backends/flashinfer/>
- vLLM FP8 KV Cache/attention 状态文章：<https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md>
- vLLM `kv_cache_dtype_skip_layers` 测试：<https://github.com/vllm-project/vllm/blob/main/tests/quantization/test_fp8.py>
- LLM Compressor KV Cache 标定示例（后续增强）：<https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache>

## 26. 审阅时需要确认的唯一事项

本规格已经把主模型、框架、镜像策略、驱动约束、搜索算法、指标、成功标准和运行边界固定下来。用户批准后进入实现计划和编码阶段；若无修改意见，批准语句为：

```text
规格已批准，开始实施。
```
