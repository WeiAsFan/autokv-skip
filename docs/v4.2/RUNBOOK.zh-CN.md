# v4.2 离线运行手册

本版复用 v4.1 已生成的数据及已运行成功的 A6000 FP4 环境，跳过 discovery、confirmation 和重新生成题目。容量目标为 2.0×，最终总分质量容差仍为 0.01。默认四路请求并发。

服务器有 Git，但不能联网或推送；Linux 登录设备能联网但不能推送。代码由 Linux 传入，精简结果经 SCP 传回 Linux，再通过 GitHub 网页手动提交。启动不要求 Git 提交、推送、doctor、smoke 或人工核对哈希。

## 1. Linux 登录设备：准备代码

从 GitHub 网页下载 `v4.2` 分支 ZIP，解压为新的 `autokv-skip-v4.2`。向服务器传入源码即可；无需重新下载模型、FlashInfer、SQuAD 或 Hotpot。

```bash
export AUTOKV_SSH='用户名@服务器地址'
scp -r ./autokv-skip-v4.2 "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/"
```

本次代码 ZIP 可能仍含历史 `results/`，传入前可以从下载副本中省略这些历史结果以减少传输；实验不需要把它们复制到服务器。不要覆盖服务器上的 v4.1 原始运行目录或依赖。

## 2. 服务器：复用成功环境和原数据

下面路径按 v4.1 正式运行填写；如果实际目录不同，修改 `AUTOKV_V41_ROOT` 和源运行 ID。`--reuse-run` 指向原始运行目录，不是网页结果目录或精简包。

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip-v4.2
export AUTOKV_V41_ROOT=/mnt_d/huangxiaoyuan/autokv-skip-v4.1
export AUTOKV_V41_RUN="$AUTOKV_V41_ROOT/runs/v41-40ef467cef6a32d2"
export AUTOKV_VLLM_BIN=/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/vllm
export AUTOKV_MODEL_PATH='/已有模型目录/Mistral-7B-Instruct-v0.3'
export AUTOKV_PYTHON="$(dirname "$AUTOKV_VLLM_BIN")/python"
export AUTOKV_FP4_SITE="$AUTOKV_V41_ROOT/.runtime/v41/site"
mkdir -p runs/_environment
cp "$AUTOKV_V41_RUN/inputs/runtime.json" runs/_environment/lock.json
```

这里复制的是成功运行保存的环境设置，包含原 CUDA 12.9、TVM-FFI 兼容路径、FlashInfer/cubin/torch 编译缓存位置。v4.2 已保留这些设置的支持，继续使用对应旧目录；不重新安装或覆盖依赖。`AUTOKV_VLLM_BIN` 和 `AUTOKV_MODEL_PATH` 覆盖环境记录中的可执行文件和模型路径。

如果在别的 shell 启动，重新设置以上环境变量。若旧成功运行需要额外 shell 环境（例如动态库路径），继续沿用那些设置。先停止占用同一 GPU 的旧实验，避免显存与计时互相影响。

直接导入的文件为源运行目录下：

```text
inputs/data/v4.1/v41-40ef467cef6a32d2/quality/experiment.jsonl
inputs/data/v4.1/v41-40ef467cef6a32d2/quality/test.jsonl
inputs/data/v4.1/v41-40ef467cef6a32d2/quality/dataset-manifest.json
construction/result.json
run-manifest.json
```

这 512 条实验题和 4096 条测试题原样复制到新运行输入目录。不需要 `--source-dir`，也不会调用数据生成器。原构造报告仅作为历史依据。数据与配置不匹配时直接说明缺失或不一致的文件，不能自动换一批题绕过。

## 3. 正式运行与续跑

```bash
set -o pipefail
bash scripts/run-v42.sh --reuse-run "$AUTOKV_V41_RUN" --json \
  2>&1 | tee runs/v42-cli.log
printf '%s\n' "${PIPESTATUS[0]}" > runs/v42-cli.exitcode
```

日志会打印 `v42-...` 运行 ID。流程为：重新评估 P32/P0 → 固定名额束搜索 → 冻结候选 → 4096 题独立测试。不会重跑 discovery/confirmation。不要使用 `--construction-only`。

中断后使用实际 ID 续跑：

```bash
bash scripts/run-v42.sh --run-id v42-实际运行ID --json \
  2>&1 | tee -a runs/v42-cli.log
printf '%s\n' "${PIPESTATUS[0]}" > runs/v42-cli.exitcode
```

续跑只补缺失回答，重放固定排序不会扩大完整评估名额。输入已经复制到新运行内，续跑不再读取旧题库，但仍须保留所引用的成功运行环境。修改代码、数据或运行配置后应开始新运行，不能把不同环境回答混用。

默认配置 `configs/v4.2/quality.json`：每深度最多 8 个候选补到 128 题，全程最多 6 个混合候选补到 512 题，其中提前确认最多 3 个。最多保留 9 层 BF16。搜索正常请求上界 24320，硬预算 24576 包含重试；端点和独立测试另计。独立测试最多 12288 次请求，需要为它留出运行时间。

候选生成与初筛仍会重建服务，不能仅按完整评估数量估算耗时。`complete=true` 只表示流程按规则结束；`technical_goal_passed=true` 才表示独立测试质量与实测容量同时达标。`no_feasible_within_budget` 只能解释为本次预算内未找到。测试失败不自动更换候选。

结果在 `runs/v42-实际运行ID/`：

- `report/QUALITY-v4.zh-CN.md`：正文标明 v4.2、质量、容量、预算与完成深度。
- `report/DATA-REUSE.zh-CN.md`：数据来源；`SOURCE-v4.1-CONSTRUCTION-...` 是历史确认报告。
- `completed-manifest.json`、`selection.json`、`search-trace.jsonl`：完成状态与搜索证据。
- `policies/`、`inputs/`：服务器保留的原始回答、完整日志、长提示和实际输入。

## 4. 服务器：生成小型分析包

成功、负结果、失败和中断都可导出，不要求主目标通过。

```bash
"$AUTOKV_PYTHON" -m scripts.export_v42_results --run-id v42-实际运行ID
```

如果在创建运行目录前失败，省略 `--run-id`，导出已经保存的 `runs/v42-cli.log` 和退出码。省略 ID 时会导出新项目内全部 v42 运行。转换旧 v4.1 结果也可显式提供 `v41-...` ID 与 `--project-root`。

命令输出 `results/v42-实际时间/`，包含：中文说明、已有报告、每个运行一个 `*-evidence.zip`，以及 CLI 日志摘要。ZIP 内是精简 JSON/JSONL 和必要实际代码，**不是原始归档**，不产生 `.tar.gz.part-*`。它保留逐题真值与短回答、分数、配对 ID、搜索决策、容量、请求数、耗时及错误摘要，可复核主要结论。

保留服务器的 `runs/v42-.../` 和原 v4.1 运行目录。小型分析包不包含长提示全文、完整服务日志或模型；需要进一步检查提示构造或完整故障现场时，再定向回传对应文件。不要删除原始目录来换取上传空间。

## 5. Linux 登录设备：回传并从 GitHub 网页提交

将导出器打印的实际目录名代入：

```bash
scp -r "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/autokv-skip-v4.2/results/v42-实际时间" ./
```

在 Linux 浏览器进入仓库，选择 **v4.2** 分支，进入 `results/`，使用 **Add file → Upload files** 上传刚接收的整个结果目录，保留文件名和层级。上传说明、报告、`evidence.zip` 和存在的 CLI 摘要，填写中文说明后在网页手动提交。**不上传原始归档、旧分片、整个 runs 或模型文件。** 两台设备均不执行 `git push`。

下载后用 Python 标准库即可解压分析：

```bash
python3 -m zipfile -e v42-实际运行ID-evidence.zip analysis
```

`export-summary.json` 记录每个策略各阶段的回答数量及坏行位置。报告与逐题数据一同保留，即使只有部分结果也能判断已完成范围，不能把尚未执行的测试写成通过。
