# AutoKV-Skip

AutoKV-Skip 是一个面向 AI 算法工程师面试的小型 vLLM 推理优化项目。它自动选择部分层保留 BF16 KV，其余层使用 FP8 E4M3 KV；在独立留出集的质量下降约束内，争取相同 KV 显存预算下至少 `1.5×` 的实测容量。不修改 vLLM/CUDA 内核，不量化模型权重。

当前分支为 **v2.1**，从 [v2.1 离线运行手册](docs/v2.1/RUNBOOK.zh-CN.md) 开始；设计改动见 [v2.1 五项修正说明](docs/v2.1/CORRECTIONS.zh-CN.md)，目标定义见 [共同语言](CONTEXT.md)。v1.0 的历史结果以原始产物和 [v1.0 统一项目事实](docs/v1.0/FACTS.zh-CN.md) 为准。

## v2.1 实现状态

- 修正变量与值配对评分、BF16 基础题无效却判为通过的问题。
- 保留已通过的组和单层配置，允许 `P1`；同层集合复用结果，从已评估合格候选中选择较小 BF16 预算。
- 主验收为质量分差与至少 `1.5×` 实测 KV 容量；`complete` 与 `technical_goal_passed` 分开。
- 留出集报告包含逐任务分数和配对区间；三组随机对照改为可选的 `v2-random-controls`，性能测量另行开展。
- `v2-run` 首次使用服务器已有 tokenizer 离线生成 `data/v2.1/quality/`；自然 QA 复用旧数据已选的 12 条样本，不再下载 LongBench。新数据共 45 条，原 v2.0 数据保留不变。

当前完成的是实现与开发验证，尚无 v2.1 正式 GPU 结果。真实质量和容量必须由目标服务器测量。

小型服务器有 Git，但不能联网或推送；Linux 登录设备能联网，也不能推送。结果和失败日志通过 `python -m scripts.export_v2_results` 导出，经 `scp` 回传 Linux 设备后，从 GitHub 网页上传并手动提交。主实验不要求 doctor、smoke、预先提交或推送。

## v1.0 当前结论

修复后的 full 运行 `8181c9a332ef6e9c` 已完成真实 vLLM/GPU 验证：

- 模型为 `mistralai/Mistral-7B-Instruct-v0.3`，GPU 为 RTX A6000 48 GiB；
- 运行环境为本地 vLLM Python 环境，驱动 `580.173.02`、CUDA `12.9`、vLLM `0.1.dev19475+gc18d29d36`；
- BF16、全 FP8、Auto-4 以及 8 个同预算/位置对照均完成 45 个质量样本；
- Auto-4 选择层为 `[1, 4, 8, 12]`；
- 三个主配置在 6 组 input/output 长度上各重复 3 次，共完成 2700 个 serving 请求，失败 0 个；
- 结果归档 SHA-256 和归档内 268 个受清单覆盖的产物哈希均已复核。

### 质量与容量摘要

| 配置 | EM | 平均 Q | KV token capacity | 相对 BF16 |
|---|---:|---:|---:|---:|
| BF16 | 1.0 | 0.9951674671 | 131072 | 1.0000× |
| 全 FP8 | 1.0 | 0.9951287091 | 262144 | 2.0000× |
| Auto-4 | 1.0 | 0.9951613370 | 232992 | 1.777588× |

BF16 与全 FP8 的平均 Q 差只有 `3.8758e-5`，配对 95% CI 为 `[-8.175e-5, 1.548e-4]`，远未达到预注册质量缺口阈值 `0.01`。Auto-4 相对全 FP8 的平均 Q 差为 `3.2628e-5`，置信区间同样跨 0；Auto-4 相对每样本 Random-4 中位数的置信区间也跨 0。

因此，v1.0 能支持的结论是：

- 逐层混合 KV dtype 的部署链路和容量模型已验证；
- 在当前数据上，全 FP8 与 BF16 的质量不可分辨，没有 Auto-4 可以恢复的质量缺口；
- Auto-4 提供了符合理论的 1.777588× 容量，但没有证明优于容量更高的全 FP8，也没有证明自动选层优于同预算随机选层；
- 这是有意义的工程负结果，不能改指标或挑样本把它包装成正结果。

### 性能结果的限制

本次 Auto-4 相对 BF16 的请求吞吐变化在 6 个场景中为 `-0.861%` 到 `+19.607%`，相对全 FP8 大多慢约 `1.2%–3.9%`。但是本次运行开启了 prefix caching，最高命中率约 50%，且同一 server 内的后续重复明显受缓存热身影响。因此这些性能数值只作为暂定证据；正式性能结论必须在关闭 prefix caching 后重跑。

## v1.0 可复现性状态

结果已在 GitHub 提交 `096697c647603aaf871f1babf3d2e8b1ceb37c1b` 中归档。结果 manifest 记录的运行源码提交为 `2c6a4e1ac96e9835a6af2ad450cb9dfa269e4953`；其 20 项受控源码已逐项按 manifest 校验，并以 `b2bb7759e687cdb61fc4ea750ac4bb74ae593f6a` 合入 `main`。合入后源码树 SHA-256 仍为 `64ad8dcaf4f36cf9f3b9575a0a324b656d3ed06f314907226ca9709154542c70`，与运行 manifest 完全一致。

原始运行提交号与 `main` 上的发布提交号不同，但受控源码内容和树 hash 相同；复现实验时应使用上述 `main` 发布提交，并同时保留这个映射事实。

## v2.0 历史状态

`v2.0` 已实现“质量缺口驱动的混合精度自动选择”：45 样本三层 Quality v2、可选 BF16-only pilot、P32/P0 端点判断、条件式 8 组/8 单层搜索、P2/P4/P8 早停、held-out 和 3 个同预算随机对照。所有 v2 server 显式关闭 prefix caching，保留实际 dtype 与服务端 prompt token 检查。

45 条 v2.0 数据已在服务器生成，并随 `9ac0341` 纳入仓库；目前没有可供分析的 v2.0 正式 GPU 结果。v2.1 保留这份历史数据，并按新的评分定义另建数据版本。

旧版操作记录见 [v2.0 离线实验手册](docs/v2.0/RUNBOOK.zh-CN.md)，只适用于 `v2.0` 分支。当前分支使用上面的 v2.1 手册。

## 项目边界

- 真实实验不在当前本机执行。操作者先在另一台 Linux 设备上使用 `ssh` 登录小型服务器，再在该服务器上运行项目。
- 当前本地工作区用于代码、文档和结果分析，不能替代服务器上的 GPU 验证。
- v1.0 固定输出 Auto-4；“根据质量缺口自动选择混合精度”是 v2.0 的正式方向，不追溯改写 v1.0。
- 项目不是论文级算法首创，也不是生产级在线服务；其价值是展示 vLLM 部署、KV Cache/量化理解、故障定位、实验设计、性能分析和诚实决策能力。

## 文档导航

- [v2.1 五项修正说明](docs/v2.1/CORRECTIONS.zh-CN.md)：本版验收标准、评分、候选保留和可选分析。
- [v2.1 离线运行手册](docs/v2.1/RUNBOOK.zh-CN.md)：服务器运行与 Linux 设备网页归档。

- [v1.0 统一项目事实](docs/v1.0/FACTS.zh-CN.md)：本次运行环境、矩阵、指标、结果和限制的唯一统一说明。
- [v1.0 对应源码发布要求](docs/v1.0/SOURCE-PUBLICATION-REQUIREMENT.zh-CN.md)：把结果与精确源码提交闭环的必做事项。
- [v2.0 设计文档](docs/v2.0/DESIGN.zh-CN.md)：质量缺口驱动的 `P_k` 自动选择、三层数据与验证规则。
- [v2.0 阶段 2–4 执行计划](docs/v2.0/EXECUTION-PLAN.zh-CN.md)：规模受控的实现、GPU 运行和精简证据计划。
- [v2.0 离线实验手册](docs/v2.0/RUNBOOK.zh-CN.md)：传入源码、正式运行、保存失败日志、传回结果与网页上传。
- [v1.0 运行前服务器手册](RUNBOOK.zh-CN.md)：历史操作方案，不是本次运行的精确复现指南。
- [v1.0 运行前技术规格](docs/superpowers/specs/2026-08-24-autokv-skip-design.md)：历史设计意图。
- [v1.0 运行前实现计划](docs/superpowers/plans/2026-08-24-autokv-skip-implementation.md)：历史实现计划。
- [推理优化技术综述](docs/research/inference-optimization-landscape.zh-CN.md)：研究背景及运行前技术判断。
- [v1.0 面试讲解稿](docs/interview/AutoKV-Skip-interview-guide.zh-CN.md)：运行前模板，尚未按本次结果完成改写。

## License

仓库代码用于个人学习、实验与面试展示。模型、vLLM、FlashInfer 及引用论文各自遵循其原始许可证。
