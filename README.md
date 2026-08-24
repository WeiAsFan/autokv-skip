# AutoKV-Skip

AutoKV-Skip 是一个面向单张 NVIDIA RTX A6000 48 GiB 的小型 vLLM 推理优化项目。它不改 vLLM/CUDA 内核，也不量化模型权重；它利用 vLLM 已有的逐层 KV Cache 量化跳过参数，自动挑出 4 个对 FP8 KV 量化更敏感的层保留 BF16，其余 28 层使用 FP8 E4M3。

项目的目标不是声称首创混合精度 KV 算法，而是用少量标准库 Python 代码完成一个面试中可讲清、服务器上可复现、结果可能为负也仍然有效的算法—系统闭环：

1. 固定驱动、镜像 digest、模型 revision、数据 hash 和随机种子；
2. 用 NIAH 长上下文召回任务估计层敏感度；
3. quick 模式以 coarse-to-fine 搜索把核心配置从 34 个减到 18 个；
4. 与 BF16、全 FP8、5 个 Random-4、First-4、Last-4、Inverted-4 公平对比；
5. 按相同输入/输出长度输出质量、KV token capacity、吞吐、TTFT、`nvidia-smi dmon` 遥测、CSV 和 SVG 报告。

正式流程还会在双启动 smoke 中全局锁定 NLL 或编辑距离评分模式；Docker pull 最多指数退避尝试 3 次，请求超时只重启同一配置且最多 2 次，避免瞬时故障改变实验条件。每次 server 启动后还会核对 Docker 中的实际 argv，并在 DEBUG 日志里逐层证明 Auto-4 的 4 层为原生 KV dtype、其余 28 层为 `fp8_e4m3`，而不是只相信传入参数。

## 冻结实验

| 项目 | 固定值 |
|---|---|
| GPU | NVIDIA RTX A6000 48 GiB，SM 8.6 |
| 宿主驱动 | `535.230.02`，严禁脚本升级、降级或重装 |
| 框架 | vLLM |
| 主镜像 | `vllm/vllm-openai:v0.26.0` |
| 兼容回退 | `vllm/vllm-openai:v0.19.1`，只在 CUDA/feature gate 失败时尝试 |
| 模型 | `mistralai/Mistral-7B-Instruct-v0.3`，revision 运行时锁为 40 位 commit |
| 模型计算 | BF16 |
| 注意力后端 | `FLASHINFER` |
| KV 预算 | 每配置固定 `16G` |
| Auto-4 | 4 层 BF16 KV + 28 层 FP8 E4M3 KV |
| Scale MVP | 固定 seed 的 `--calculate-kv-scales`；两次独立 smoke 必须一致 |

Mistral 的 32 层、8 KV heads、head dimension 128 对应：

```text
BF16 KV = 2(K/V) × 32 × 8 × 128 × 2 bytes = 131072 bytes/token
FP8  KV = 2(K/V) × 32 × 8 × 128 × 1 byte  =  65536 bytes/token
Auto-4  = 4 × 4096 + 28 × 2048             =  73728 bytes/token
Auto-4 理论容量 / BF16 = 131072 / 73728     = 1.7778×
```

## 当前验证状态

- **本地代码验证**：标准库单元测试、伪 HTTP 服务、伪 Docker 生命周期、恢复状态、命令转义、报告渲染、quick/full dry-run 和静态安全扫描可在无 GPU 的机器完成。
- **目标服务器尚未验证**：真实 R535 forward compatibility、目标镜像拉取、FlashInfer + FP8、逐层 skip、真实 KV capacity 和 quick 实验数值，必须等能够访问 A6000 服务器后按运行手册执行。
- 因此，本仓库中的代码通过不等于 GPU 实验已经成功。任何报告数值都只能来自 `runs/<run-id>/` 的真实产物，不能预填或臆造。

## 从哪里开始

首次上服务器时不要直接运行一键实验。逐项执行 [RUNBOOK.zh-CN.md](RUNBOOK.zh-CN.md)，它包含阶段 0–13、每条可复制命令、预计时间、成功判据和失败分支。

先在任意 Python 3.10+ 环境执行本地验证：

```bash
python3 scripts/verify.py
```

在 Windows 本地开发环境也可以运行：`python scripts/verify.py`。

然后在目标 Linux 服务器按顺序执行的核心入口是：

```bash
python3 -m autokv doctor --profile quick --json
python3 -m autokv lock-image --profile quick --json
python3 -m autokv make-data --profile quick --json
python3 -m autokv dry-run --profile quick --json
python3 -m autokv smoke --profile quick --json
python3 -m autokv run --profile quick --json
```

`run` 可断点续跑。完成且 SHA-256、行数、上下文、实际 command/server log 和状态文件一致的配置不会重跑；selection 每次会从 probe 证据重新推导，benchmark 只有在 matrix state 同时锚定全部 raw JSON 与 dmon 时才复用。控制器只管理带有 `io.autokv.project=autokv-skip` label 的精确容器；即使遥测停止失败或 `docker run -d` 超时，也会先核验所有权再清理。主机重启留下的同名 `Exited` 容器会在核验 label 后自动移除；同名容器仍运行或属于其他项目时则拒绝操作。

若某个产物被手工改坏，控制器不会“洗白”它。先保留诊断，再从命令记录文件名取得 12 位配置 ID，仅对相应阶段执行 `--force CONFIG_ID`；旧证据会移动到可恢复的 `_superseded/`，不会删除。完整命令见运行手册阶段 11。

## 产物

```text
configs/                         冻结 quick/full profiles
data/niah/                       确定性 NIAH 规格与 dataset hash
runs/_environment/doctor.json   宿主与容器 gate 证据
runs/_environment/lock.json     镜像 RepoDigest、模型 revision、版本锁
runs/<run-id>/run-manifest.json profile/data/image/model/源码树 SHA-256 与 Git 身份
runs/<run-id>/probe/             组/层敏感度原始 JSONL
runs/<run-id>/selection.json     Auto/随机/首尾/反向层集合
runs/<run-id>/quality/           11 个配置的最终质量原始 JSONL
runs/<run-id>/perf/              3 个主配置的容量、bench JSON、dmon 与 matrix state
runs/<run-id>/report/            中文 Markdown、总表 CSV、逐场景 CSV、SVG
runs/<run-id>/completed-manifest.json 全部活跃 run 产物的最终哈希清单
runs/<run-id>/_superseded/       `--force` 前移入的可恢复旧证据
runs/_diagnostics/               脱敏诊断包；不含模型和 HF cache
```

## 阅读路径

- [完整服务器操作手册](RUNBOOK.zh-CN.md)
- [推理优化技术综述](docs/research/inference-optimization-landscape.zh-CN.md)
- [面试讲解稿](docs/interview/AutoKV-Skip-interview-guide.zh-CN.md)
- [批准的技术规格](docs/superpowers/specs/2026-08-24-autokv-skip-design.md)
- [实现计划](docs/superpowers/plans/2026-08-24-autokv-skip-implementation.md)

## 诚实的成功定义

预注册主验收条件是 FP8 capacity 至少为 BF16 的 1.85×、Auto-4 至少为 1.65×；只有当 `Q_bf16 - Q_fp8 > 0.01` 时才计算 gap recovery，并要求 Auto-4 回收至少 50%。Auto-4 还应高于 Random-4 中位数和 Inverted-4。

若质量没有可测缺口，结论是“FP8 在本任务上已足够”；若 Auto-4 不优于随机，结论是“这个边际敏感度代理在该模型/任务上没有证明有效”。这两种都是可讲的工程结果，禁止事后更换指标或筛选运行。

## License

仓库代码用于个人学习、实验与面试展示。模型、vLLM、FlashInfer 及引用论文各自遵循其原始许可证。
