# AutoKV-Skip v4.0 离线运行手册

适用分支：`v4.0`。默认主命令完成“难度扫描 → 两批新数据确认 → 冻结生成规则 → 新实验集/测试集 → 自动选层 → 独立测试”。代码和 CPU 验证已完成；本地验证不代表已经找到 BF16 可解、FP8 有恢复需求的数据，也不代表 GPU 主实验达标。

服务器有 Git，但不能访问外网或推送 GitHub。Linux 登录设备能访问外网，也不能推送 GitHub。文件由 Linux 设备传入服务器，结果由服务器传回 Linux，最后在 Linux 的 GitHub 网页上手动上传并提交。启动实验不要求 Git 提交、推送、doctor、smoke 或人工核对哈希。

## 1. Linux 登录设备：准备代码

以下在联网 Linux 登录设备的 Bash 执行，填写实际 SSH 地址。命令从已经发布的 `v4.0` 分支下载源码 ZIP。

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_TRANSFER="$(mktemp -d "$HOME/autokv-v40-transfer.XXXXXX")"
curl --fail --location \
  'https://github.com/WeiAsFan/autokv-skip/archive/refs/heads/v4.0.zip' \
  --output "$AUTOKV_TRANSFER/source.zip"
unzip -q "$AUTOKV_TRANSFER/source.zip" -d "$AUTOKV_TRANSFER/source"
export AUTOKV_SOURCE="$(find "$AUTOKV_TRANSFER/source" -mindepth 1 -maxdepth 1 -type d -print -quit)"
cd "$AUTOKV_SOURCE"
tar -czf "$AUTOKV_TRANSFER/autokv-v40-source.tar.gz" \
  autokv scripts configs pyproject.toml README.md CONTEXT.md \
  docs/v4.0 data/v4.0/README.md data/v3.0/README.md
scp "$AUTOKV_TRANSFER/autokv-v40-source.tar.gz" "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/"
```

## 2. Linux 登录设备：只在缺少离线来源时准备

v4.0 复用 v3.0 的 SQuAD/Hotpot 四份原始 JSON 和 `source-manifest.json`。服务器已有完整来源目录时，跳过本节下载和传输，后面将 `AUTOKV_DATA_SOURCE` 指向它即可。

没有完整来源包时，在 Linux 登录设备运行既有准备脚本。只使用 Python 标准库，无需模型、GPU 环境或完整评测框架。

```bash
cd "$AUTOKV_SOURCE"
python3 -m scripts.prepare_v3_sources \
  --output "$AUTOKV_TRANSFER/data-source" --hotpot-source huggingface
```

已有原始文件时，用下面的离线导入代替下载。文件名为 `train-v2.0.json`、`dev-v2.0.json`、`hotpot_train_v1.1.json`、`hotpot_dev_distractor_v1.json`。

```bash
python3 -m scripts.prepare_v3_sources \
  --input-dir '/已有原始JSON目录' --output "$AUTOKV_TRANSFER/data-source"
```

准备成功后再打包传输；下载中断可重复同一准备命令继续。

```bash
tar -czf "$AUTOKV_TRANSFER/autokv-v40-data-source.tar.gz" \
  -C "$AUTOKV_TRANSFER/data-source" .
scp "$AUTOKV_TRANSFER/autokv-v40-data-source.tar.gz" "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/"
```

## 3. 小型服务器：使用已有环境

从 Linux 设备执行 `ssh "$AUTOKV_SSH"` 登录。以下命令在服务器 Bash 执行。使用独立项目目录，不覆盖正在运行的旧实验。

```bash
export AUTOKV_ROOT='/mnt_d/huangxiaoyuan/autokv-skip-v4.0'
mkdir -p "$AUTOKV_ROOT"
tar -xzf /mnt_d/huangxiaoyuan/autokv-v40-source.tar.gz -C "$AUTOKV_ROOT"
cd "$AUTOKV_ROOT"

export AUTOKV_VLLM_PYTHON='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/python'
export AUTOKV_VLLM_BIN='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/vllm'
export AUTOKV_MODEL_PATH='/mnt_d/huangxiaoyuan/autokv-skip/.cache/huggingface/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71'
export AUTOKV_PORT=8010
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

模型目录按已有快照真实位置填写；路径包含 `huggingface/hub/` 时只修改该变量，不重新下载。复用 v3.0 来源时设置：

```bash
export AUTOKV_DATA_SOURCE='/mnt_d/huangxiaoyuan/autokv-skip-v3.0/data/v3.0/source'
```

若前面传入了新来源包，则改用以下命令：

```bash
export AUTOKV_DATA_SOURCE="$AUTOKV_ROOT/data/v4.0/source"
mkdir -p "$AUTOKV_DATA_SOURCE"
tar -xzf /mnt_d/huangxiaoyuan/autokv-v40-data-source.tar.gz -C "$AUTOKV_DATA_SOURCE"
```

复用已有环境记录中的 CUDA/FlashInfer 路径；显式设置的模型和 vLLM 路径优先。没有环境记录也可用上述变量运行。

```bash
mkdir -p runs/_environment
if [ ! -f runs/_environment/lock.json ]; then
  for AUTOKV_ENV_SOURCE in \
    /mnt_d/huangxiaoyuan/autokv-skip-v3.0/runs/_environment/lock.json \
    /mnt_d/huangxiaoyuan/autokv-skip-v2.1/runs/_environment/lock.json \
    /mnt_d/huangxiaoyuan/autokv-skip/runs/_environment/lock.json; do
    if [ -f "$AUTOKV_ENV_SOURCE" ]; then
      cp "$AUTOKV_ENV_SOURCE" runs/_environment/lock.json
      break
    fi
  done
fi
tmux new -s autokv-v40
```

tmux 新会话继承上述变量；重新登录后新开会话时重新设置变量。不升级驱动或重装 vLLM。端口被占用时换一个空闲端口。

## 4. 小型服务器：运行完整实验

默认构造条件为 BF16 正确率至少 `0.80`、P32−P0 **严格大于 `0.10`**。需要更强缺口时，在首次运行前修改 `configs/v4.0/quality.json` 的 `construction.min_fp8_gap` 为 `0.20`。正式质量容差仍是总分 `0.01`、每任务 `0.02`，实测 KV 容量目标 `1.5×`。

以下是默认主命令，不需先生成数据或运行开发检查：

```bash
cd "$AUTOKV_ROOT"
mkdir -p runs
AUTOKV_ATTEMPT="v4-run-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
"$AUTOKV_VLLM_PYTHON" -u -m autokv v4-run \
  --project-root "$AUTOKV_ROOT" --source-dir "$AUTOKV_DATA_SOURCE" \
  --port "$AUTOKV_PORT" --json \
  2> >(tee "runs/$AUTOKV_ATTEMPT.log" >&2) \
  | tee "runs/$AUTOKV_ATTEMPT.json"
AUTOKV_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$AUTOKV_EXIT" > "runs/$AUTOKV_ATTEMPT.exitcode"
```

运行 ID 会在构造开始前写入 stderr 日志，形式为 `v4-...`。默认顺序如下：

1. 生成 24 个预定条件各 64 题。完整证据放不下的条件保留原因，不发送请求。所有可构造题按精度合并评估 P32/P0。
2. 自动选最多两个条件，每条件生成两批各 512 个新实例。两批分别满足可解性和强缺口才确认成功；不按模型对错筛题，也不追加批次直到通过。
3. 没有成功条件时输出 `construction_not_found`，本轮结束。成功则自动冻结一个生成规则，生成新实验集 512 题、测试集 4096 题。
4. 在新实验集上比较 P32/P0，必要时执行 `32→96→512` 多保真束搜索和反向删层，先写入 `selection.json`，再运行独立测试。

如果新实验集的 P0 已满足最终质量容差，正常保留 P0 早停结论。没有可行候选时不会强测失败策略。程序不自动降低构造阈值或改成 FP4。

## 5. 只做构造与中断续跑

只想先取得构造报告时，可在主命令中增加 `--construction-only`，其他日志保存方式相同。这是可选停止位置，不是默认流程的前置步骤。此时 `stage=construction`、`complete=true` 只表示构造阶段结束。

接续正式阶段或恢复中断时，从日志复制运行 ID，使用下列命令。显式 `--run-id` 读取该轮保存的配置；当前配置文件修改不会悄悄覆盖旧轮。

```bash
export AUTOKV_RUN_ID='v4-替换为日志中的实际ID'
AUTOKV_ATTEMPT="v4-resume-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
"$AUTOKV_VLLM_PYTHON" -u -m autokv v4-run \
  --project-root "$AUTOKV_ROOT" --run-id "$AUTOKV_RUN_ID" \
  --source-dir "$AUTOKV_DATA_SOURCE" --port "$AUTOKV_PORT" --json \
  2> >(tee "runs/$AUTOKV_ATTEMPT.log" >&2) \
  | tee "runs/$AUTOKV_ATTEMPT.json"
AUTOKV_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$AUTOKV_EXIT" > "runs/$AUTOKV_ATTEMPT.exitcode"
```

同一轮逐题续跑，保留发现/确认结果、生成种子、搜索预算和已冻结层集合；正常错误答案不重测。请求暂时失败最多自动重试一次，不当作零分。已生成输入可直接复用；还要补生成题时需要原来源包，可以通过 `--source-dir` 指定其新位置。

按 `Ctrl-b`、`d` 脱离 tmux；`tmux attach -t autokv-v40` 返回。中断后等该轮旧进程退出，再执行续跑命令。一次只启动一个命令使用该运行目录。

源码、模型、实际运行环境或来源实质改变后不能混用旧回答；恢复原输入以续跑，或不指定旧 `--run-id` 开始新一轮。识别过程由程序完成，不要求人工比较摘要。改变种子、提示、难度或阈值是新实验；看过的测试题不能重新声称独立，本工作区内跨 v3/v4 的使用记录会标明重用。

## 6. 预算与结果判断

默认发现最多 3072 次正常回答，确认最多 4096 次，两阶段理想共 4 次启动。记录结构不适配或只确认一个条件时实际更少。正式端点 1024 次；选层/删层预算 24576 次实际请求（含重试）；独立测试最多 12288 次正常回答。全链最多 45056 次正常回答，端点/确认/测试的异常重试另计。具体耗时取决于长度、启动和搜索晋级，不能根据 CPU 验证推算 GPU 时限。

`search.max_seconds` 默认 `null`；需限制搜索时间时在新轮运行前设置秒数。预算在请求边界检查。达到预算后只冻结已完整评估的可行候选，不把部分分数当成功。

| 字段或状态 | 解释 |
|---|---|
| `construction_passed` | 两批新确认均满足 `BF16 ≥ 0.80` 和严格超过预定 FP8 缺口 |
| `construction_not_found` | 该有限扫描与确认范围未找到；是可导出的负结果，不证明所有数据都不存在 |
| `quality_passed` | 已冻结候选在测试点估计满足最终质量容差 |
| `technical_goal_passed` | 有效独立测试保质，且真实 KV 容量至少 `1.5×` |
| `data_conditions_reproduced` | 正式测试再次满足构造的数据条件；未运行测试为 `null` |
| `recovery_demonstrated` | 构造通过、测试复现强缺口、主目标达标，且候选为非端点混合配置 |
| `no_feasible_within_budget` | 搜索范围/预算内无完整可行候选，未进行测试 |
| `test_quality_failed` | 实验集可行但测试质量失败，不能据此改层后重复测试 |
| `capacity_unverified` / `capacity_below_target` | 真实容量缺失/冲突或低于目标；理论容量不能替代 |
| `test_not_independent` | 检测到其他运行使用过同题或同证据，只算诊断 |
| `runtime_failed` / `interrupted` | 执行未完成，查看日志后续跑 |

退出码 0、`complete=true` 不等于优化成功。通过字段是预定点估计判断；配对区间单独呈现，不自动构成联合 95% 非劣性证明。重复背景的区间解释以固定背景池为条件，全配对差相同时标明 bootstrap 退化。

主要产物位于 `runs/<run_id>/`：

| 路径 | 内容 |
|---|---|
| `inputs/` | 实际源码、配置、模板、来源/划分说明及正式数据副本 |
| `construction/discovery.jsonl`、`confirmation.jsonl` | 构造实例、题内映射、背景 ID、目标证据 token 位置 |
| `construction/conditions.jsonl`、`result.json` | 全部条件状态、逐批/合并统计、结构失败原因 |
| `construction/selected-rule.json` | 成功后冻结的生成规则和确认依据 |
| `policies/*/{discovery,confirmation,experiment,test}.jsonl` | 各阶段原始回答、分数、耗时、输出截断等 |
| `policies/*/attempts/` | 每次服务启动、实际命令、日志、请求计数和容量 |
| `search-trace.jsonl`、`selection.json` | 选层过程与测试前冻结的候选 |
| `report/CONSTRUCTION-v4.zh-CN.md` | 构造报告，包括负结果与中断信息 |
| `report/QUALITY-v4.zh-CN.md` | 正式阶段质量/容量报告 |
| `completed-manifest.json` | 当前阶段、完成/达标字段与累计开销 |

正式数据也复制到 `data/v4.0/<run_id>/quality/` 方便查看；续跑以运行目录内的实际输入为准。

## 7. 小型服务器：无论成功、负结果或失败都导出

先将日志中的运行 ID 填入 `AUTOKV_RUN_ID`，导出单轮全部阶段：

```bash
export AUTOKV_RUN_ID='v4-替换为日志中的实际ID'
"$AUTOKV_VLLM_PYTHON" -m scripts.export_v4_results \
  --project-root "$AUTOKV_ROOT" --run-id "$AUTOKV_RUN_ID"
```

首次完整命令结束后若尚未设置 `AUTOKV_RUN_ID`，可从 CLI JSON 读取；命令失败且 JSON 不完整时直接复制 stderr 中的 ID。

```bash
export AUTOKV_RUN_ID="$("$AUTOKV_VLLM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "runs/$AUTOKV_ATTEMPT.json")"
```

还没有建立运行目录、只有 CLI 失败日志，或希望打包全部本版运行时，省略 `--run-id`：

```bash
"$AUTOKV_VLLM_PYTHON" -m scripts.export_v4_results --project-root "$AUTOKV_ROOT"
```

命令输出 `results/v4-时间戳/` 路径。目录包含 Markdown 报告、上传说明和归档；大归档自动拆为不超过 20 MiB 的分片。不打包模型权重、整个原始来源库或 `.git`。归档保留构造和正式阶段全部实际输入、原始答案、部分回答、错误及服务/CLI 日志。

## 8. Linux 登录设备：接收结果并从 GitHub 网页提交

退出服务器或另开 Linux 登录设备终端，设置实际导出目录名，执行 SCP：

```bash
export AUTOKV_RESULT_DIR='v4-替换为导出目录的实际时间戳'
mkdir -p "$AUTOKV_TRANSFER/results"
scp -r "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/autokv-skip-v4.0/results/$AUTOKV_RESULT_DIR" \
  "$AUTOKV_TRANSFER/results/"
```

在 Linux 浏览器打开 [GitHub 的 v4.0 分支](https://github.com/WeiAsFan/autokv-skip/tree/v4.0)，进入 `results/`，选择 **Add file → Upload files**，上传刚接收的整个 `v4-时间戳` 文件夹。确认分片、README 和报告均在该文件夹下，填写提交说明，如“上传 v4.0 数据构造与选层实验结果”，通过网页提交到 `v4.0`。

单次选择文件过多时，仍按同一目录分批上传；所有 `.part-*` 必须传齐。不要只上传成功布尔值或截图。两台 Linux 设备都不运行 `git push`。解包方式已写在每个导出目录的 `README.zh-CN.md` 中。
