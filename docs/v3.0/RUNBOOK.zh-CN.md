# AutoKV-Skip v3.0 离线运行手册

适用分支：`v3.0`。目标是用 256 条实验数据选择层集合，再用 1024 条独立测试数据验证质量和至少 `1.5×` 的实测 KV 容量。代码与 CPU 开发验证已经完成，真实 GPU 结果需按本手册在目标服务器取得。

小型服务器有 Git，但不能访问外网或推送 GitHub。Linux 登录设备能访问外网，也不能推送 GitHub。源码和数据由 Linux 设备传入服务器；结果及失败日志传回 Linux 后，从 GitHub 网页上传并手动提交。启动实验不要求先提交、推送、运行 doctor 或 smoke，也不要求人工核验哈希。

## 1. Linux 登录设备：准备源码

以下命令均在 Linux 设备的 Bash 执行，先填入实际 SSH 地址。GitHub 下载方式适用于实现已发布到远端 `v3.0` 的情况；若使用开发工作区提供的源码 ZIP，把它放到同一 `source.zip` 路径并跳过 `curl`，其余步骤相同，不需要 Git 提交。

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_TRANSFER="$(mktemp -d "$HOME/autokv-v30-transfer.XXXXXX")"
curl --fail --location \
  'https://github.com/WeiAsFan/autokv-skip/archive/refs/heads/v3.0.zip' \
  --output "$AUTOKV_TRANSFER/source.zip"
unzip -q "$AUTOKV_TRANSFER/source.zip" -d "$AUTOKV_TRANSFER/source"
export AUTOKV_SOURCE="$(find "$AUTOKV_TRANSFER/source" -mindepth 1 -maxdepth 1 -type d -print -quit)"
cd "$AUTOKV_SOURCE"
tar -czf "$AUTOKV_TRANSFER/autokv-v30-source.tar.gz" \
  autokv scripts configs pyproject.toml README.md CONTEXT.md \
  docs/v3.0 data/v3.0/README.md
```

原有 `v2-run` 继续服务历史版本。正式 v3.0 实验只使用本手册的 `v3-*` 命令。

## 2. Linux 登录设备：准备来源并传入服务器

准备脚本只用 Python 标准库，不需要下载模型、安装 GPU 环境或完整 RULER 框架。原始来源总量为数百 MB。下例使用已核查结构的 HotpotQA 公开镜像；省略 `--hotpot-source` 会先尝试原站，失败时自动使用镜像。

```bash
cd "$AUTOKV_SOURCE"
python3 -m scripts.prepare_v3_sources \
  --output "$AUTOKV_TRANSFER/data-source" --hotpot-source huggingface
```

脚本成功后再打包。下载失败时保留镜像页缓存，重复同一命令继续；不要把没有成功生成来源清单的半成品传入服务器。

已有完整原始 JSON 时，可以用下列命令替代下载，不会访问外网：

```bash
python3 -m scripts.prepare_v3_sources \
  --input-dir '/已有原始JSON所在目录' --output "$AUTOKV_TRANSFER/data-source"
```

四个文件名为 `train-v2.0.json`、`dev-v2.0.json`、`hotpot_train_v1.1.json`、`hotpot_dev_distractor_v1.json`。后续传输仍在 Linux 设备执行：

```bash
tar -czf "$AUTOKV_TRANSFER/autokv-v30-data-source.tar.gz" \
  -C "$AUTOKV_TRANSFER/data-source" .
scp "$AUTOKV_TRANSFER/autokv-v30-source.tar.gz" \
  "$AUTOKV_TRANSFER/autokv-v30-data-source.tar.gz" \
  "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/"
ssh "$AUTOKV_SSH"
```

## 3. 小型服务器：独立目录与已有环境

从这里开始在服务器 Bash 执行。首次部署使用独立目录；更新已有目录前让其正在运行的实验结束，不覆盖活动实验的源码和输入。

```bash
export AUTOKV_ROOT='/mnt_d/huangxiaoyuan/autokv-skip-v3.0'
mkdir -p "$AUTOKV_ROOT/data/v3.0/source"
tar -xzf /mnt_d/huangxiaoyuan/autokv-v30-source.tar.gz -C "$AUTOKV_ROOT"
tar -xzf /mnt_d/huangxiaoyuan/autokv-v30-data-source.tar.gz \
  -C "$AUTOKV_ROOT/data/v3.0/source"
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

模型路径以服务器已有快照的真实位置为准；若实际路径包含 `huggingface/hub/`，只改路径变量，不重新下载模型。也可以把旧项目的 `runs/_environment/lock.json` 复制到本项目相同位置，继续复用其中的 CUDA/FlashInfer 路径设置；上面的两个显式路径变量会覆盖锁中的模型与 vLLM 路径。

```bash
mkdir -p runs/_environment
if [ ! -f runs/_environment/lock.json ]; then
  for AUTOKV_ENV_SOURCE in \
    /mnt_d/huangxiaoyuan/autokv-skip-v2.1/runs/_environment/lock.json \
    /mnt_d/huangxiaoyuan/autokv-skip-v2.0/runs/_environment/lock.json \
    /mnt_d/huangxiaoyuan/autokv-skip/runs/_environment/lock.json; do
    if [ -f "$AUTOKV_ENV_SOURCE" ]; then
      cp "$AUTOKV_ENV_SOURCE" runs/_environment/lock.json
      break
    fi
  done
fi
tmux new -s autokv-v30
```

tmux 新会话通常继承上述环境变量；如果重新登录后进入新会话，应重新设置这些路径变量。端口被其他服务使用时换一个空闲端口，不终止其他人的服务。不升级驱动或重装现有 vLLM。

## 4. 小型服务器：一次性开发评估

首次确定新数据协议时，在独立开发样本上只运行 BF16。先离线生成开发数据：

```bash
cd "$AUTOKV_ROOT"
mkdir -p runs
set -o pipefail
"$AUTOKV_VLLM_PYTHON" -u -m autokv v3-make-data \
  --project-root "$AUTOKV_ROOT" --mode development \
  --source-dir data/v3.0/source --json \
  2> >(tee runs/v3-make-development.log >&2) | tee runs/v3-make-development.json

"$AUTOKV_VLLM_PYTHON" -u -m autokv v3-run \
  --project-root "$AUTOKV_ROOT" --development --port "$AUTOKV_PORT" --json \
  2> >(tee runs/v3-development.log >&2) | tee runs/v3-development.json
```

开发报告路径在输出 JSON 的 `report` 字段。观察四任务的可解性、逐长度表现、空答案和输出截断。QA 得分约 `0.6–0.9` 只是难度参考，检索接近满分不自动失效；不能按逐题 BF16/FP8 对错筛选正式题。

如需调整提示结构或输出预算，在 `configs/v3.0/quality.json` 或对应生成代码修改并记录理由，再使用新的数据版本；已有数据不会被程序静默覆盖。调整时可将本项目 `data/v3.0/development/` 重命名为带日期的历史目录后再生成，旧运行输入副本保留。开发样本不进入正式测试。

开发评估不是每次运行都要通过的前置程序。数据定义已经确定、已有合适开发记录时直接进入正式数据准备。

## 5. 小型服务器：正式数据与主实验

确认开发阶段的数据定义后，生成正式 256/1024 数据。此步骤仅用本地 tokenizer，通常比 GPU 实验便宜，但处理完整自然语料与长提示仍需要时间；进度会按任务和长度输出。

```bash
"$AUTOKV_VLLM_PYTHON" -u -m autokv v3-make-data \
  --project-root "$AUTOKV_ROOT" --mode formal \
  --source-dir data/v3.0/source --json \
  2> >(tee runs/v3-make-formal.log >&2) | tee runs/v3-make-formal.json
```

生成成功后运行主命令，记录本次 CLI 输出与退出码：

```bash
AUTOKV_ATTEMPT="v3-run-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
"$AUTOKV_VLLM_PYTHON" -u -m autokv v3-run \
  --project-root "$AUTOKV_ROOT" --port "$AUTOKV_PORT" --json \
  2> >(tee "runs/$AUTOKV_ATTEMPT.log" >&2) \
  | tee "runs/$AUTOKV_ATTEMPT.json"
AUTOKV_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$AUTOKV_EXIT" > "runs/$AUTOKV_ATTEMPT.exitcode"
```

顺序是 P32/P0 全实验集 → 必要时多保真束搜索和删层 → 自动冻结层集合 → P32/P0/候选独立测试 → 质量与容量报告。候选之间不共享 KV 张量。补样本只补缺少的回答，32→96→256 不会把前面已经完成的题全部重跑。

按 `Ctrl-b`、`d` 脱离 tmux；`tmux attach -t autokv-v30` 返回。中断后确认旧进程已退出，使用相同源码、配置、数据和环境重复主命令，逐题续跑。正常错误答案不会重试；暂时性 HTTP 错误最多自动重试一次。

不要依据测试结果改层、改阈值或改提示后继续宣称独立测试通过。更换代码或配置会产生新的运行 ID，但本项目能识别工作区中已用测试问题/证据的重用，并标记为 `test_not_independent`。这类重跑只能用于诊断；新一轮主结论需要新的独立测试，且应披露历史使用。

## 6. 成本与结果判断

P0 若直接满足条件，主流程为 2560 次回答、4 次启动。若第一轮 P1 成功且候选差异足够清楚，示例为 5440 次回答、约 47 次启动；用旧运行成本粗估约 6.3 小时。并列与不确定候选会扩大评估，搜索较深也会明显增加成本，不能把这个示例当作时限。

配置中的 `search.max_requests` 和 `search.max_seconds` 默认为 `null`，可在运行前设置选层与删层的资源预算；端点和独立测试另计。墙钟预算在请求边界检查，正在执行的请求会结束后再检查。`search.schedule="full"` 可直接评估候选的完整实验集，用于启动成本很高的环境；同一次实验不要中途更换调度来挑结果。

| 状态 | 解释 |
|---|---|
| `passed` | 参照有效、独立测试点估计满足约束、真实容量至少 `1.5×` |
| `test_quality_failed` | 实验集可行，但测试质量未达标 |
| `reference_degenerate` | BF16 某主任务全零，不能用两边都为零宣称保留质量 |
| `no_feasible_within_budget` | 在当前层数或资源预算内未找到，不代表不存在解 |
| `capacity_unverified` / `capacity_below_target` | 缺少一致的实测容量，或容量没有达标 |
| `test_not_independent` | 其他运行已经使用同一问题或证据，本次是诊断重跑 |
| `runtime_failed` / `interrupted` | 流程未完成，查看日志并续跑 |
| `development_complete` | 仅完成独立开发样本的 BF16 评估，不是主实验达标 |

退出码 0、`complete=true` 只表示流程已得到终局结论。主目标看 `technical_goal_passed=true`。报告同时提供配对区间；区间上界仍超出容差时，只能陈述点估计达标，不能宣称已经证明 1% 非劣性。

主要文件在 `runs/<run_id>/`：实际输入在 `inputs/`，每题回答与各次启动日志在 `policies/`，选择过程在 `search-trace.jsonl`，测试前冻结结果在 `selection.json`，总结在 `report/QUALITY-v3.zh-CN.md`。硬中断导致部分启动没有结束时间时，报告会说明时间统计只是已记录的下界。

## 7. 小型服务器：成功或失败都导出

命令成功时，从 CLI JSON 的 `run_id` 找到本次运行并导出：

```bash
AUTOKV_RUN_ID="$("$AUTOKV_VLLM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "runs/$AUTOKV_ATTEMPT.json")"
"$AUTOKV_VLLM_PYTHON" -m scripts.export_v3_results \
  --project-root "$AUTOKV_ROOT" --run-id "$AUTOKV_RUN_ID"
```

失败或还没建立运行目录时，直接导出可见的全部 v3 结果及 CLI 日志：

```bash
"$AUTOKV_VLLM_PYTHON" -m scripts.export_v3_results --project-root "$AUTOKV_ROOT"
```

导出器打印 `results/v3-<UTC时间>/` 的实际目录。归档包括原始回答、失败现场、每次服务日志、搜索轨迹及运行输入，不要求成功报告或完成清单。大于约 20 MiB 时自动分片，说明文件包含还原命令。

## 8. Linux 登录设备：回传并网页提交

退出 SSH 或另开 Linux 终端，设置真实服务器地址和上一步打印的导出目录：

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_EXPORT_NAME='v3-替换为导出目录的实际UTC时间'
mkdir -p "$HOME/autokv-v30-results"
scp -r "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/autokv-skip-v3.0/results/$AUTOKV_EXPORT_NAME" \
  "$HOME/autokv-v30-results/"
```

在 Linux 浏览器进入 GitHub 仓库并选择 `v3.0` 分支，打开 `results/`，使用 **Add file → Upload files** 上传该次导出目录中的说明、报告、归档或全部分片，填写中文提交说明并在网页手动提交。不要上传原始语料库、模型缓存或 `.git`。服务器与 Linux 登录设备均不执行 `git push`。
