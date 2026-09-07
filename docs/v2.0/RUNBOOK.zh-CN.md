# AutoKV-Skip v2.0 离线实验手册

适用分支：`v2.0`。本手册覆盖阶段 2–4 的质量实验，阶段 5 性能实验另行安排。

小型服务器已安装 Git，可用于本地版本管理，但无法访问外网或向 GitHub 仓库推送。用于 SSH 登录的 Linux 设备可以访问外网，但也无法向 GitHub 推送；它负责下载代码、接收服务器回传的运行结果和日志，再通过该设备上的 GitHub 网页端上传并手动提交。实验流程不要求两台设备执行 `git commit` 或 `git push`。

当前仓库已包含在服务器生成的 45 条正式数据：calibration 27 条、held-out 18 条，难度为 `standard`。这次继续实验直接使用它们，跳过 LongBench 下载、数据生成和 pilot。`9ac0341` 只补交了数据；本次修复还需要更新源码。

## 1. Linux 设备：准备并传入源码

下面的命令在 **Linux 登录设备的 Bash** 中执行。先设置实际 SSH 地址：

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_TRANSFER="$(mktemp -d "$HOME/autokv-transfer.XXXXXX")"
```

如果修复已通过 GitHub 网页上传到 `v2.0`，下载该分支：

```bash
curl --fail --location \
  'https://github.com/WeiAsFan/autokv-skip/archive/refs/heads/v2.0.zip' \
  --output "$AUTOKV_TRANSFER/source.zip"
```

如果修复尚未上传 GitHub，可直接使用本次交付的 `autokv-skip-v2-offline.zip`：

```bash
cp "$HOME/Downloads/autokv-skip-v2-offline.zip" "$AUTOKV_TRANSFER/source.zip"
```

以上两种来源选一种。解压后只传本次实验需要的源码、文档和正式数据：

```bash
unzip -q "$AUTOKV_TRANSFER/source.zip" -d "$AUTOKV_TRANSFER/source"
AUTOKV_SOURCE="$(find "$AUTOKV_TRANSFER/source" -mindepth 1 -maxdepth 1 -type d -print -quit)"
tar -czf "$AUTOKV_TRANSFER/autokv-v2-source.tar.gz" \
  -C "$AUTOKV_SOURCE" \
  autokv scripts configs pyproject.toml README.md docs/v2.0 data/v2/quality
scp "$AUTOKV_TRANSFER/autokv-v2-source.tar.gz" \
  "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/autokv-v2-source.tar.gz"
ssh "$AUTOKV_SSH"
```

## 2. 小型服务器：更新项目并复用运行环境

以下命令在 **小型服务器的 Bash** 中执行。确认本项目当前没有实验正在运行后再更新源码。原来的 `runs/`、环境锁和已冻结数据会保留。

```bash
export AUTOKV_ROOT='/mnt_d/huangxiaoyuan/autokv-skip-v2.0'
mkdir -p "$AUTOKV_ROOT"
if [ -f "$AUTOKV_ROOT/data/v2/quality/dataset-manifest.json" ]; then
  tar -xzf /mnt_d/huangxiaoyuan/autokv-v2-source.tar.gz \
    -C "$AUTOKV_ROOT" --exclude='data/v2/quality'
else
  tar -xzf /mnt_d/huangxiaoyuan/autokv-v2-source.tar.gz -C "$AUTOKV_ROOT"
fi
cd "$AUTOKV_ROOT"
export AUTOKV_VLLM_PYTHON='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/python'
mkdir -p runs/_environment
if [ ! -f runs/_environment/lock.json ] && \
   [ -f /mnt_d/huangxiaoyuan/autokv-skip/runs/_environment/lock.json ]; then
  cp /mnt_d/huangxiaoyuan/autokv-skip/runs/_environment/lock.json runs/_environment/lock.json
fi
```

程序从已有 `runs/_environment/lock.json` 读取 vLLM、模型路径和环境变量，不要求重新生成 doctor 或 lock 记录，也不核对旧驱动版本号。模型直接使用锁中的 `model_path`，无需复制到新项目的 `.cache`。

如果没有可复用的锁，设置下面两个路径即可。`AUTOKV_MODEL_PATH` 应指向已下载的模型快照目录；本次模型 revision 为 `c170c708c41dac9275d15a8fff4eca08d52bab71`。

```bash
export AUTOKV_VLLM_BIN='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/vllm'
export AUTOKV_MODEL_PATH='/替换为服务器上已有的模型快照目录'
```

已有环境锁时不要执行这段占位路径示例。运行期间保持 vLLM 环境、模型和配置不变；若主动更换环境，使用新项目目录运行，避免复用旧结果。

## 3. 小型服务器：执行正式实验并保存诊断

选择有资源的 GPU 和空闲端口。之前的记录表明 `8000` 被其他服务占用，下面示例使用 `8010`；若它也被占用，改用其他端口。程序会在启动时检查端口，只管理自己启动的 vLLM 进程。

不需要执行 `scripts/verify.py`、doctor、lock-image、dry-run 或 smoke，也不需要先把冻结数据发布到 GitHub。测试属于开发验证，不是服务器实验的前置步骤。

```bash
tmux new -s autokv-v2
```

在新建的 tmux 会话内执行；如果第 2 节采用无环境锁的方式，先在该会话内重新设置 `AUTOKV_VLLM_BIN` 和 `AUTOKV_MODEL_PATH`：

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip-v2.0
export AUTOKV_ROOT="$(pwd -P)"
export AUTOKV_VLLM_PYTHON='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/python'
export AUTOKV_PORT=8010
mkdir -p runs
AUTOKV_ATTEMPT="v2-run-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
"$AUTOKV_VLLM_PYTHON" -u -m autokv v2-run \
  --project-root "$AUTOKV_ROOT" \
  --port "$AUTOKV_PORT" \
  --json \
  2> >(tee "runs/$AUTOKV_ATTEMPT.log" >&2) \
  | tee "runs/$AUTOKV_ATTEMPT.json"
AUTOKV_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$AUTOKV_EXIT" > "runs/$AUTOKV_ATTEMPT.exitcode"
printf '运行退出码：%s；诊断日志：runs/%s.log\n' "$AUTOKV_EXIT" "$AUTOKV_ATTEMPT"
```

标准输出 JSON、标准错误日志和退出码按尝试分别保存。失败也执行第 5 节导出，不要只记录成功时的 JSON。进度日志会说明正在运行或复用的策略。

按 `Ctrl-b`，再按 `d` 脱离 tmux。重新登录后用 `tmux attach -t autokv-v2` 返回。不要同时启动第二个 `v2-run`。

### 自动执行顺序

1. `P32`、`P0` 各运行 27 条 calibration。
2. 若 P0 满足质量约束，直接验证两个端点的 held-out。
3. 若有质量缺口，运行 8 个四层组、前两组中的 8 个单层，再依次测试 `P2 → P4 → P8`，第一个合格预算立即早停。
4. 中间策略入选时，held-out 运行两个端点、所选策略和 3 个同预算随机策略。
5. 输出决策、质量报告和完成清单。

| 分支 | 正式启动数 | 正式样本请求数 |
|---|---:|---:|
| 无质量缺口 | 4 | 90 |
| 有缺口且 P2 通过 | 25 | 621 |
| 有缺口且 P4 通过 | 26 | 648 |
| 有缺口且 P8 通过 | 27 | 675 |
| P8 仍失败并回退 P32 | 23 | 603 |

暂时性 HTTP 故障最多重试一次，不会额外运行 smoke 或增加候选预算。

### 中断后继续

确认旧实验进程已经退出，再重复正式命令。源码、配置、数据和运行环境身份相同时，完整策略直接复用；未完成或原始结果损坏的策略移入 `_incomplete/` 后重跑。

Git 提交号、工作区是否干净、日志排版和日志 hash 不参与恢复判断。运行前的实际源码、配置、数据和环境路径保存在 `runs/<run-id>/inputs/`，随后与结果一起导出。改变运行代码或输入会生成新运行 ID。

## 4. 小型服务器：阅读结果

退出码 `0` 且 CLI JSON 的 `complete` 为 `true` 表示本次编排完成。在同一 shell 中查看：

```bash
AUTOKV_RUN_ID="$("$AUTOKV_VLLM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "runs/$AUTOKV_ATTEMPT.json")"
cat "runs/$AUTOKV_RUN_ID/decision.json"
cat "runs/$AUTOKV_RUN_ID/selection.json"
cat "runs/$AUTOKV_RUN_ID/report/QUALITY-v2.zh-CN.md"
```

四种结论都有效：

- `final.k = 0`：全 FP8 满足 calibration 和 held-out 约束。
- `final.k = 2/4/8` 且 `layer_selection_supported=true`：预算和层排序得到本次证据支持。
- `final.k = 2/4/8` 且 `layer_selection_supported=false`：支持混合预算，未证明排序优于同预算随机选择。
- `final.k = 32`：中间预算不合格或未泛化，回退 BF16。

这些结论不代表吞吐、TTFT、TPOT 或 ITL 有提升。日志缺少容量行时，实测容量记录为 `null`，不因此丢弃质量结果。

## 5. 小型服务器：导出结果和日志

实验成功、失败或中断后都可导出。先等本次进程退出，再执行：

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip-v2.0
export AUTOKV_ROOT="$(pwd -P)"
export AUTOKV_VLLM_PYTHON='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/python'
AUTOKV_EXPORT="$("$AUTOKV_VLLM_PYTHON" -m scripts.export_v2_results --project-root "$AUTOKV_ROOT")"
printf '结果导出目录：%s\n' "$AUTOKV_EXPORT"
ls -lh "$AUTOKV_EXPORT"
```

默认导出全部 v2 运行；只需某一次时，在命令后加 `--run-id "$AUTOKV_RUN_ID"`。启动阶段失败、还没有 run manifest 时，也能导出 CLI 日志。

导出目录为 `results/v2-<UTC时间>/`，包含中文说明、已有质量报告和压缩归档。归档包含原始 JSONL、server 日志、未完成策略、每次 CLI 输出和输入副本，不收集模型、`.cache/`、`.git/` 或完整 LongBench 源数据目录。

超过 20 MiB 的归档自动分片，每片最多 20 MiB。无需在服务器运行 Git 或逐文件哈希校验。

## 6. Linux 设备：接收文件并从 GitHub 网页上传

回到 **Linux 登录设备**，把上一节打印的导出目录最后一段填入 `AUTOKV_EXPORT_NAME`：

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_ROOT='/mnt_d/huangxiaoyuan/autokv-skip-v2.0'
export AUTOKV_EXPORT_NAME='v2-替换为上一步输出的UTC时间'
mkdir -p "$HOME/autokv-results"
scp -r "$AUTOKV_SSH:$AUTOKV_ROOT/results/$AUTOKV_EXPORT_NAME" "$HOME/autokv-results/"
ls -lh "$HOME/autokv-results/$AUTOKV_EXPORT_NAME"
```

在 Linux 设备的浏览器操作：

1. 打开 [v2.0 的 results 目录](https://github.com/WeiAsFan/autokv-skip/tree/v2.0/results)，确认分支为 `v2.0`。
2. 选择 **Add file → Upload files**，拖入本机 `autokv-results/` 下整个 `v2-<UTC时间>` 文件夹，保留目录层级。
3. 上传说明、报告、归档或全部分片；提交说明写“归档 v2 质量实验结果与日志”，点击 **Commit changes**。失败诊断可写“归档 v2 实验失败日志”。
4. 网页刷新后确认文件都在，保存目录链接供后续分析。

GitHub 网页单文件上限为 25 MiB，一次最多上传 100 个文件；20 MiB 分片留有余量。文件较多时分批上传到同一目录。见 [GitHub 官方上传说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository?platform=linux)。

在 Linux 设备解包查看：

```bash
cd "$HOME/autokv-results/$AUTOKV_EXPORT_NAME"
# 只有分片归档才执行下面这一行：
cat autokv-v2.tar.gz.part-* > autokv-v2.tar.gz
tar -xzf autokv-v2.tar.gz
```

网页上传归档、分片、说明和报告；解压出的目录用于本地分析。

## 7. 保留的检查与故障处理

| 现象 | 处理 |
|---|---|
| 端口不可用 | 改 `--port`，不停止其他人的服务 |
| 本地模型目录不存在 | 修正锁中的 `model_path` 或 `AUTOKV_MODEL_PATH` |
| vLLM 不识别 dtype/skip-layers/prefix-caching 参数 | 查看 server 日志，使用支持项目参数的已有 runtime |
| 实际 dtype 或混合层日志与策略不符 | 保留日志并修正 runtime；错误精度的结果不能用于比较 |
| 日志明确显示 prefix caching 开启 | 修正启动行为；缺少该日志文字本身不阻断实验 |
| 服务端 `prompt_tokens` 与数据值不同 | 修正模型/tokenizer/chat template，保证各策略使用相同输入 |
| 数据内容或配置与 manifest 不符 | 恢复已冻结配置与数据；CRLF/LF 差异已自动兼容 |
| 回答含替换字符、重复片段或答错 | 保存输出并正常评分，不用首条回答决定能否运行整个策略 |
| 单策略结果不完整或被改动 | 重跑正式命令，由程序只重跑受影响策略 |
| held-out 不合格 | 按规则回退 P32，不换样本、不用 held-out 重排层 |

冻结数据、模型、评分、端点判断、候选预算和 held-out 隔离规则保持原样。哈希只用于识别实际输入与防止复用损坏的原始结果，不要求手工计算，也不检查 GitHub 发布状态。

### 仅在开展新数据实验时

本次已有正式数据，不执行本段。新实验应使用独立项目目录，在可联网的 Linux 设备准备源数据并传入服务器，然后使用服务器已有模型的 tokenizer 执行 `v2-freeze-data`。可选的 BF16-only pilot 最多 9 请求，只能在正式数据生成前运行；数据已存在时禁止事后调整难度。已生成的数据不需要先提交 GitHub 就能开始实验。
