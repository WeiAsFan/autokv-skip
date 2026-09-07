# AutoKV-Skip v2.1 离线运行手册

适用分支：`v2.1`。主实验验证质量下降阈值与至少 `1.5×` 的实测 KV 容量收益。代码沿用 `v2-run` 等命令名，默认配置已切换为 `configs/v2.1/quality.json`。

小型服务器有 Git，但不能访问外网或推送 GitHub。用于 SSH 登录的 Linux 设备可以访问外网，也不能推送 GitHub。源码经 Linux 设备下载后传入服务器，结果和失败日志传回 Linux 设备，再从 GitHub 网页上传并手动提交。发布不是启动实验的前置条件。

## 1. Linux 登录设备：下载并传入源码

在 Linux 设备的 Bash 中设置真实 SSH 地址：

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_TRANSFER="$(mktemp -d "$HOME/autokv-v21-transfer.XXXXXX")"
curl --fail --location \
  'https://github.com/WeiAsFan/autokv-skip/archive/refs/heads/v2.1.zip' \
  --output "$AUTOKV_TRANSFER/source.zip"
unzip -q "$AUTOKV_TRANSFER/source.zip" -d "$AUTOKV_TRANSFER/source"
AUTOKV_SOURCE="$(find "$AUTOKV_TRANSFER/source" -mindepth 1 -maxdepth 1 -type d -print -quit)"
tar -czf "$AUTOKV_TRANSFER/autokv-v21-source.tar.gz" \
  -C "$AUTOKV_SOURCE" autokv scripts configs pyproject.toml README.md CONTEXT.md \
  docs/v2.1 data/v2/quality data/v2.1
scp "$AUTOKV_TRANSFER/autokv-v21-source.tar.gz" \
  "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/autokv-v21-source.tar.gz"
ssh "$AUTOKV_SSH"
```

也可在 Linux 设备浏览器下载 `v2.1` 分支 ZIP，再将下载文件放到上述 `source.zip` 路径，从解压步骤继续。

## 2. 小型服务器：使用独立项目目录和已有环境

在服务器的 Bash 执行。更新已有 v2.1 目录时，先确认该目录的旧实验已退出。

```bash
export AUTOKV_ROOT='/mnt_d/huangxiaoyuan/autokv-skip-v2.1'
mkdir -p "$AUTOKV_ROOT"
tar -xzf /mnt_d/huangxiaoyuan/autokv-v21-source.tar.gz \
  -C "$AUTOKV_ROOT" --exclude='data/v2.1/quality'
cd "$AUTOKV_ROOT"
mkdir -p runs/_environment
if [ ! -f runs/_environment/lock.json ]; then
  for AUTOKV_ENV_SOURCE in \
    /mnt_d/huangxiaoyuan/autokv-skip-v2.0/runs/_environment/lock.json \
    /mnt_d/huangxiaoyuan/autokv-skip/runs/_environment/lock.json; do
    if [ -f "$AUTOKV_ENV_SOURCE" ]; then
      cp "$AUTOKV_ENV_SOURCE" runs/_environment/lock.json
      break
    fi
  done
fi
```

已有锁只用于读取 vLLM、模型和必要环境路径，不要求重做 doctor 或核对旧驱动版本号。不复制模型、不重新安装运行环境，不运行旧版多阶段手册。

没有可用环境锁时，设置真实路径。下列模型路径必须替换为已有模型快照目录，模型 revision 为 `c170c708c41dac9275d15a8fff4eca08d52bab71`：

```bash
export AUTOKV_VLLM_BIN='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/vllm'
export AUTOKV_MODEL_PATH='/替换为服务器已有的Mistral模型快照目录'
```

已有正确锁时跳过这段占位示例。运行期间保持模型、配置和 vLLM 环境不变；主动更换环境时使用新的项目目录，避免把旧结果作为新环境结果复用。

## 3. 小型服务器：一条命令运行主实验并保存日志

选择有资源的 GPU 和空闲端口。之前 `8000` 被其他服务占用，示例使用 `8010`；若仍占用，修改端口，不终止其他人的服务。

```bash
tmux new -s autokv-v21
```

在 tmux 会话内执行。使用无锁方式时，在此会话内重新设置上一节两个路径变量。

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip-v2.1
export AUTOKV_ROOT="$(pwd -P)"
export AUTOKV_VLLM_PYTHON='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/python'
export AUTOKV_PORT=8010
mkdir -p runs
AUTOKV_ATTEMPT="v2-run-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
"$AUTOKV_VLLM_PYTHON" -u -m autokv v2-run \
  --project-root "$AUTOKV_ROOT" --port "$AUTOKV_PORT" --json \
  2> >(tee "runs/$AUTOKV_ATTEMPT.log" >&2) \
  | tee "runs/$AUTOKV_ATTEMPT.json"
AUTOKV_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$AUTOKV_EXIT" > "runs/$AUTOKV_ATTEMPT.exitcode"
printf '退出码：%s；日志：runs/%s.log\n' "$AUTOKV_EXIT" "$AUTOKV_ATTEMPT"
```

首次运行自动离线生成 `data/v2.1/quality/`：使用现有 tokenizer 生成新合成题，复用旧数据中已选定的自然 QA，然后开始实验。此步骤不启动 GPU 服务、不访问外网，也不需要完整 LongBench 源目录。以后直接复用新数据；不要把旧 v2.0 JSONL 手工复制到新数据目录。

正式执行顺序为：BF16 基础题有效性 → P0 质量判断 → 有缺口时八组/八层搜索并保留合格候选 → 独立 held-out → 质量及实测容量报告。不会额外运行 doctor、smoke 或推送检查。配置与数据必须一致，实际 dtype、prompt token 和结果完整性仍由程序检查。

| 路径 | GPU 服务启动 | 正式样本请求 |
|---|---:|---:|
| Calibration 的 BF16 基础题失效 | 1 | 27 |
| P0 在两个 split 均达标 | 4 | 90 |
| 有缺口且 P1 达标 | 21 | 540 |
| 有缺口且 P2 达标 | 22 | 567 |
| 主实验最坏路径上限 | 24 | 621 |
| 可选三组随机对照 | 额外 3 | 额外 54 |

同层集合去重可进一步减少启动数。上表不含中断重跑或暂时性 HTTP 故障的一次重试；没有强行凑足启动次数的步骤。

按 `Ctrl-b` 再按 `d` 脱离 tmux，使用 `tmux attach -t autokv-v21` 返回。不要同时启动两个主实验。中断后确认旧进程退出，再重复主实验命令；完整策略自动复用，未完成策略归入 `_incomplete/` 后重跑。

## 4. 读懂结果

退出码 `0`、`complete=true` 只表示流程已给出结果。**主验收看 `technical_goal_passed=true`**：留出集质量达标且实测 KV 容量至少 `1.5×`。

主命令输出完整 JSON 后，可在同一 shell 查看报告：

```bash
AUTOKV_RUN_ID="$("$AUTOKV_VLLM_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "runs/$AUTOKV_ATTEMPT.json")"
cat "runs/$AUTOKV_RUN_ID/report/QUALITY-v2.zh-CN.md"
```

| `status` | 解释 |
|---|---|
| `passed` | 质量与实测容量均达标，P0 或中间混合配置都可能出现 |
| `invalid_reference` | BF16 基础题失效，不能判断候选质量；查看已保存的原始回答和日志 |
| `heldout_failed` | 候选未泛化，回退 P32，本次技术目标未达成 |
| `no_qualifying_mixed_policy` | 已评估候选均不合格，回退 P32，没有容量收益 |
| `capacity_unverified` | 质量已过，但缺少实测容量；保留质量结果并检查原始启动日志 |
| `capacity_below_target` | 质量已过，实测容量未达到目标 |

报告提供任务绝对分数、配对差和区间、质量阈值判定、理论及实测容量。置信区间用于说明不确定性；45 条样本不能用于声称普遍无损。即使留出集上的 P0 也达标，也不据此重新选择已确定的混合候选。

## 5. 可选：同预算随机对照

主实验完成后，只有通过 held-out 的中间候选才需要这项分析。P0/P32 自动跳过。它不影响主验收，不要求先跑性能。

```bash
AUTOKV_CONTROL_ATTEMPT="v2-controls-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
"$AUTOKV_VLLM_PYTHON" -u -m autokv v2-random-controls \
  --project-root "$AUTOKV_ROOT" --port "$AUTOKV_PORT" --json \
  2> >(tee "runs/$AUTOKV_CONTROL_ATTEMPT.log" >&2) \
  | tee "runs/$AUTOKV_CONTROL_ATTEMPT.json"
AUTOKV_CONTROL_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$AUTOKV_CONTROL_EXIT" > "runs/$AUTOKV_CONTROL_ATTEMPT.exitcode"
```

结果位于 `random-controls.json` 和 `report/RANDOM-v2.zh-CN.md`。可选分析失败时，主实验结果仍然保留；一并导出失败日志即可。高于三组随机中位数只是本次观察，不自动等于层排序有效。

## 6. 小型服务器：导出成功结果或失败诊断

先等本次进程退出。即使启动阶段失败、还没有运行目录，也执行导出：

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip-v2.1
export AUTOKV_ROOT="$(pwd -P)"
export AUTOKV_VLLM_PYTHON='/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/python'
AUTOKV_EXPORT="$("$AUTOKV_VLLM_PYTHON" -m scripts.export_v2_results --project-root "$AUTOKV_ROOT")"
printf '导出目录：%s\n' "$AUTOKV_EXPORT"
ls -lh "$AUTOKV_EXPORT"
```

导出到 `results/v2-<UTC时间>/`，包含说明、主报告、已有随机报告、源码/配置/新旧数据、运行时输入副本、原始 JSONL 和日志。运行当时的输入以 `runs/<run-id>/inputs/` 为准。不会收集模型或完整 LongBench 源库；超过 20 MiB 自动分片。无需人工哈希，也不要求本地 Git 提交。

## 7. Linux 登录设备：接收并从 GitHub 网页提交

回到 Linux 设备，把导出目录名称填入下列变量：

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_ROOT='/mnt_d/huangxiaoyuan/autokv-skip-v2.1'
export AUTOKV_EXPORT_NAME='v2-替换为上一节打印的UTC时间'
mkdir -p "$HOME/autokv-results"
scp -r "$AUTOKV_SSH:$AUTOKV_ROOT/results/$AUTOKV_EXPORT_NAME" "$HOME/autokv-results/"
ls -lh "$HOME/autokv-results/$AUTOKV_EXPORT_NAME"
```

在 Linux 设备浏览器打开 [v2.1 的 results 目录](https://github.com/WeiAsFan/autokv-skip/tree/v2.1/results)，确认分支为 `v2.1`。选择 **Add file → Upload files**，上传整个导出目录，保留目录层级。上传说明、报告和归档或全部分片后，用中文提交说明手动 **Commit changes**。不要在服务器或 Linux 登录设备执行 `git push`。

上传完成后保存该结果目录的链接。需要本地解包时，在导出目录执行：

```bash
# 仅分片归档需要这一行：
cat autokv-v2.tar.gz.part-* > autokv-v2.tar.gz
tar -xzf autokv-v2.tar.gz
```

## 8. 故障处理边界

端口占用时换端口；路径错误时修正已有模型或 vLLM 路径；实际 dtype、tokenizer 输入或配置不一致时，保留日志并修正对应输入。普通答错、重复片段等保存后正常评分；BF16 基础题失效和候选 held-out 失败通过报告说明，不再启动一串诊断实验。

吞吐、TTFT、TPOT/ITL 的后续性能实验单独安排，不反向修改质量阈值、数据或已确定的层集合。
