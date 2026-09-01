# AutoKV-Skip FP8 乱码修复与复验运行手册

> **文档状态：v1.0 bug 修复前历史手册。** 本文记录针对 2026-08-28 乱码结果的修复与复验流程，不是修复后运行 `8181c9a332ef6e9c` 的事实报告。修复后事实见 [v1.0 统一项目事实](docs/v1.0/FACTS.zh-CN.md)，运行对应源码的发布要求见 [v1.0 对应源码发布要求](docs/v1.0/SOURCE-PUBLICATION-REQUIREMENT.zh-CN.md)。

本文用于从一台 Linux 客户端通过 SSH 登录目标 A6000 服务器，保全真实运行源码，隔离 FP8 KV Cache 乱码根因，应用最小修复，并在严格正确性门禁通过后恢复 AutoKV-Skip quick 实验。

本文针对 2026-08-28 的两次已知运行：

- BF16 最终质量为 18/18 Exact Match；
- 全 FP8 和所有混合 FP8 配置均为 0 Exact Match；
- 全 FP8 在 1024-token smoke 中稳定输出 `Question:�JJJ...`；
- Auto-4 稳定输出 `1.1.1...` 等循环文本；
- 实际服务器使用本地 vLLM `0.1.dev19475+gc18d29d36`、驱动 `580.173.02`、FlashInfer `0.6.16.post3`；
- 实际运行源码为 dirty tree，且 `autokv/local_runtime.py` 尚未进入当前 GitHub 主分支。

本文不会要求升级或重装 NVIDIA 驱动，也不会删除既有 `runs/`、模型缓存或虚拟环境。

## 0. 本次操作的完成定义

只有同时满足以下条件，才可以认为“FP8 乱码 bug 已修复”：

1. 真实服务器源码已备份并提交到独立 Git 分支；
2. Git working tree 干净，运行 manifest 能记录 `git_dirty=false`；
3. BF16、全 FP8 在 256-token 和 1024-token NIAH 探针上均严格输出正确答案；
4. 全 FP8 至少经过 3 次独立服务启动，三次都正确；
5. 输出不包含替换字符 `�`，不在 96-token 上限处停止；
6. 控制器 smoke 后的独立严格检查通过；
7. probe 的 FP8 和 Auto-4 输出不是循环乱码；
8. 最终 quality 中 BF16 仍为 18/18；若 FP8 确有质量缺口，Auto-4 满足预注册恢复条件后才运行 benchmark；若 FP8 已无实质缺口，则明确记录“无需选择性保留层”的空结果；
9. 同一个新 run 中生成 `probe/`、`quality/`、`perf/`、`report/` 和 `completed-manifest.json`。

任何一步失败，都按该阶段的“失败分支”停止。不要为了得到结果而跳过门禁。

## 1. 绝对禁止事项

- 不执行 `git reset --hard`、`git clean -fd`、`rm -rf` 或覆盖式解压；
- 不在 dirty working tree 上执行 `git pull --rebase`；
- 不删除旧 `runs/`，也不把旧 perf 复制到新 quality run；
- 不使用 `pkill -f vllm`，只停止本手册记录的精确 PID；
- 不停止其他用户的 GPU 进程；
- 不升级、降级或重装 NVIDIA driver、DKMS、内核或 CUDA toolkit；
- 不把 HF Token 写入命令行、Git、日志、`.env` 或诊断包；
- 在 FP8 严格探针失败时，不运行 probe、evaluate、benchmark 或 full profile；
- 不把“重复地产生相同乱码”当成确定性成功。

## 2. 在 Linux 客户端登录服务器

以下命令在你手边的 Linux 设备上执行。只替换第一行的登录目标：

```bash
export AUTOKV_SSH_TARGET='LOGIN_NAME@SERVER_ADDRESS'

ssh \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  "$AUTOKV_SSH_TARGET"
```

如果需要指定端口：

```bash
export AUTOKV_SSH_TARGET='LOGIN_NAME@SERVER_ADDRESS'
export AUTOKV_SSH_PORT='22'

ssh \
  -p "$AUTOKV_SSH_PORT" \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  "$AUTOKV_SSH_TARGET"
```

不要把密码或 Token 写进环境变量。

## 3. 在服务器进入 tmux 并设置变量

先检查基础工具：

```bash
bash --version | head -n 1
command -v git
command -v python3
command -v curl
command -v nvidia-smi
command -v tmux
command -v ss
```

缺少 `tmux` 时先不要开始长任务。若你没有安装权限，请让管理员安装，或使用服务器已有的等价会话工具。

创建会话：

```bash
tmux new -s autokv-fp8-fix
```

若提示会话已经存在：

```bash
tmux attach -t autokv-fp8-fix
```

进入 tmux 后先单独执行下面一条，确保后续在 Bash 中运行：

```bash
bash
```

然后设置变量。以下路径来自已有运行产物；若 `test` 失败，先确认真实路径，不要猜测：

```bash
export AUTOKV_ROOT='/mnt_d/autokv-skip'
export AUTOKV_VENV='/mnt_d/autokv-skip/.venv-vllm-pgcg'
export AUTOKV_PYTHON="$AUTOKV_VENV/bin/python"
export AUTOKV_VLLM="$AUTOKV_VENV/bin/vllm"
export AUTOKV_MODEL_PATH='/mnt_d/autokv-skip/.cache/huggingface/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71'
export AUTOKV_MODEL_ID='mistralai/Mistral-7B-Instruct-v0.3'
export AUTOKV_PORT='8000'
export AUTOKV_FIX_TAG="$(date -u +%Y%m%dT%H%M%SZ)-fp8-correctness"
export AUTOKV_DIAG_DIR="$AUTOKV_ROOT/runs/_bugfix/$AUTOKV_FIX_TAG"
export AUTOKV_BACKUP_ROOT='/mnt_d/autokv-backups'

cd "$AUTOKV_ROOT"
mkdir -p "$AUTOKV_DIAG_DIR"
set +e
set -o pipefail

test -d "$AUTOKV_ROOT"
test -x "$AUTOKV_PYTHON"
test -x "$AUTOKV_VLLM"
test -d "$AUTOKV_MODEL_PATH"

printf 'ROOT=%s\nVLLM=%s\nMODEL=%s\nDIAG=%s\n' \
  "$AUTOKV_ROOT" "$AUTOKV_VLLM" "$AUTOKV_MODEL_PATH" "$AUTOKV_DIAG_DIR"
```

成功判据：四个 `test` 均无输出且退出码为 0，打印路径与服务器实际目录一致。

后续命令依赖 Bash 的 `PIPESTATUS` 和 `%q` 格式。确认当前 shell 是 Bash：

```bash
test -n "${BASH_VERSION:-}" \
  && printf 'BASH_VERSION=%s\n' "$BASH_VERSION"
```

如果这条命令没有打印版本号，先单独执行 `bash` 进入 Bash，再重新执行本节变量块。这里有意关闭 `errexit`：B 配置预期可能以退出码 10 报告正确性失败，必须先安全停止它启动的服务，再由人工读取结果；不要自行执行 `set -e`。

## 4. 只读检查：不要立即拉取或修改代码

```bash
cd "$AUTOKV_ROOT"

pwd
git status --short --branch
git remote -v
git rev-parse HEAD
git log -3 --oneline --decorate

nvidia-smi \
  --query-gpu=index,name,driver_version,memory.total,compute_cap \
  --format=csv,noheader,nounits

nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits

df -h "$AUTOKV_ROOT" "$AUTOKV_BACKUP_ROOT" 2>/dev/null || true

ss -ltnp | grep -E ":${AUTOKV_PORT}[[:space:]]" || true
```

检查要求：

- GPU 应为 `NVIDIA RTX A6000`，compute capability 为 `8.6`；
- 记录现场真实 driver，不要修改 driver；
- 若端口 8000 已被占用，先用 `ss` 输出中的 PID 和 `ps -fp PID` 确认所有者；
- 若 GPU 上有其他用户的任务，不停止、不抢占；
- 此时不要执行 `git pull`。

如果 GPU、模型目录或虚拟环境与已知环境不一致，执行第 18 节“收集诊断并停止”。

## 5. 保全 dirty 源码和未跟踪文件

生成独立备份目录：

```bash
cd "$AUTOKV_ROOT"

export AUTOKV_BACKUP_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export AUTOKV_BACKUP_DIR="$AUTOKV_BACKUP_ROOT/$AUTOKV_BACKUP_TAG"

mkdir -p "$AUTOKV_BACKUP_DIR"

git status --short --branch --untracked-files=normal \
  | tee "$AUTOKV_BACKUP_DIR/git-status.txt"

git rev-parse HEAD \
  | tee "$AUTOKV_BACKUP_DIR/git-head.txt"

git diff --binary --no-ext-diff \
  > "$AUTOKV_BACKUP_DIR/tracked-working-tree.patch"

git diff --cached --binary --no-ext-diff \
  > "$AUTOKV_BACKUP_DIR/tracked-index.patch"

git ls-files --others --exclude-standard -- \
  autokv configs tests scripts docs \
  .gitignore pyproject.toml README.md RUNBOOK.zh-CN.md \
  > "$AUTOKV_BACKUP_DIR/untracked-source-files.txt"

tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf "$AUTOKV_BACKUP_DIR/source-working-tree.tar.gz" \
  autokv configs tests scripts docs \
  .gitignore pyproject.toml README.md RUNBOOK.zh-CN.md

git bundle create "$AUTOKV_BACKUP_DIR/repository.bundle" --all

find "$AUTOKV_BACKUP_DIR" \
  -maxdepth 1 \
  -type f \
  ! -name SHA256SUMS \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | tee "$AUTOKV_BACKUP_DIR/SHA256SUMS"

find "$AUTOKV_BACKUP_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

成功判据：

- `repository.bundle` 非空；
- tracked patch 和 Git 状态已保存；
- `source-working-tree.tar.gz` 包含 tracked 与 untracked 的源码现场，但不包含 `.venv-vllm-pgcg`、模型 cache 或 `runs/`；
- `untracked-source-files.txt` 已列出 `autokv/local_runtime.py` 等源码范围内的未跟踪文件；
- `SHA256SUMS` 已生成。

不要把备份目录放进项目仓库，也不要删除它。

## 6. 将真实运行源码提交到修复分支

创建或切换到修复分支；dirty 修改会跟随到新分支：

```bash
cd "$AUTOKV_ROOT"

export AUTOKV_FIX_BRANCH="codex/fp8-correctness-$(date -u +%Y%m%d)"

if git show-ref --verify --quiet "refs/heads/$AUTOKV_FIX_BRANCH"; then
  git switch "$AUTOKV_FIX_BRANCH"
else
  git switch -c "$AUTOKV_FIX_BRANCH"
fi

git status --short --branch
git diff --stat
git status --short --untracked-files=normal
git ls-files --others --exclude-standard -- \
  autokv configs tests scripts docs \
  .gitignore pyproject.toml README.md RUNBOOK.zh-CN.md
```

如果状态中出现 `?? .venv-vllm-pgcg/`，只把这个项目专用虚拟环境目录加入忽略规则，不要暂存它：

```bash
if git status --short --untracked-files=normal \
  | grep -Fxq '?? .venv-vllm-pgcg/'; then
  grep -Fxq '.venv-vllm-pgcg/' .gitignore \
    || printf '%s\n' '.venv-vllm-pgcg/' >> .gitignore
fi

git status --short --untracked-files=normal
```

重点确认以下已知运行时文件是否存在于修改列表：

```text
autokv/benchmark.py
autokv/cli.py
autokv/commands.py
autokv/config.py
autokv/doctor.py
autokv/experiment.py
autokv/local_runtime.py
autokv/niah.py
autokv/report.py
configs/quick.json
configs/full.json
```

先阅读差异：

```bash
git diff -- autokv configs tests scripts README.md RUNBOOK.zh-CN.md
sed -n '1,260p' autokv/local_runtime.py
```

确认没有 Token、模型文件、cache、run 产物或无关文件后，显式暂存代码和对应测试：

```bash
git add \
  autokv/benchmark.py \
  autokv/cli.py \
  autokv/commands.py \
  autokv/config.py \
  autokv/doctor.py \
  autokv/experiment.py \
  autokv/local_runtime.py \
  autokv/niah.py \
  autokv/report.py \
  configs/quick.json \
  configs/full.json \
  .gitignore

git add -u -- tests scripts README.md RUNBOOK.zh-CN.md docs

# 如果上面的 untracked-source-files.txt 还列出必要的新测试或文档，
# 阅读其内容后，使用 git add -- 精确文件路径 逐个加入。

git diff --cached --check
git diff --cached --name-status
git diff --cached --stat
git diff --cached
```

如果暂存差异正确：

```bash
git commit -m "fix: record actual local vLLM runtime"
```

随后再同步远端。不要在提交前执行这一段：

```bash
git fetch origin &&
git merge --no-edit origin/main &&

python3 scripts/verify.py \
  |& tee "$AUTOKV_DIAG_DIR/verify-before-fp8-fix.log" &&

git status --short --branch &&
test -z "$(git status --porcelain)" &&
git push -u origin "$AUTOKV_FIX_BRANCH"
```

成功判据：

- 测试通过；
- `git status` 干净；
- 修复分支已推送，或至少本地 commit 和 `repository.bundle` 均已保存。

若 `git fetch` 因服务器网络或认证失败，`&&` 会阻止 merge、验证和 push 继续执行。保留本地 commit 与 bundle，记录错误；可以继续做只读环境取证和 A/B/C 最小诊断，但在恢复远端同步并重新验证前，不生成可对外宣称的正式 run。

如果 merge 冲突，不要使用 checkout/reset 丢弃任一侧。执行：

```bash
git status
git diff --name-only --diff-filter=U
```

保存输出并停止合并处理；不要进入 GPU 诊断。

## 7. 固定现场环境与 vLLM 来源

```bash
cd "$AUTOKV_ROOT"

{
  date -u --iso-8601=seconds
  uname -a
  git status --short --branch
  git rev-parse HEAD
  "$AUTOKV_PYTHON" --version
  "$AUTOKV_VLLM" --version
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits
} |& tee "$AUTOKV_DIAG_DIR/environment.txt"

PYTHONPATH="$AUTOKV_ROOT" "$AUTOKV_PYTHON" - <<'PY' | tee "$AUTOKV_DIAG_DIR/python-runtime.json"
import importlib.metadata
import json
import pathlib

import flashinfer
import torch
import vllm

dist = importlib.metadata.distribution("vllm")
payload = {
    "vllm_version": getattr(vllm, "__version__", "unknown"),
    "vllm_file": str(pathlib.Path(vllm.__file__).resolve()),
    "vllm_direct_url": dist.read_text("direct_url.json"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flashinfer": getattr(flashinfer, "__version__", "unknown"),
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

"$AUTOKV_VLLM" serve --help \
  > "$AUTOKV_DIAG_DIR/vllm-serve-help.txt" 2>&1

grep -nE \
  'attention-backend|kv-cache-dtype|calculate-kv-scales|skip-layers|prefix-caching|enforce-eager' \
  "$AUTOKV_DIAG_DIR/vllm-serve-help.txt"
```

检查 `python-runtime.json` 中的 `vllm_direct_url`。若它指向本地源码仓库，再对该源码仓库执行：

```bash
export AUTOKV_VLLM_SOURCE='/把 direct_url 中的真实源码目录填在这里'

git -C "$AUTOKV_VLLM_SOURCE" status --short --branch
git -C "$AUTOKV_VLLM_SOURCE" rev-parse HEAD
git -C "$AUTOKV_VLLM_SOURCE" diff --stat
```

如果 vLLM 源码本身也是 dirty，重复第 5 节的备份思想，保存其 commit 和 patch。不要在不知道差异的情况下重装这个虚拟环境。

## 8. 确认诊断参数和端口安全

该 vLLM 版本默认启用了 prefix caching。诊断必须显式关闭它。

```bash
if grep -q -- '--no-enable-prefix-caching' "$AUTOKV_DIAG_DIR/vllm-serve-help.txt"; then
  export AUTOKV_PREFIX_ARG='--no-enable-prefix-caching'
else
  printf '%s\n' '停止：当前 vLLM help 中没有 --no-enable-prefix-caching' >&2
  printf '%s\n' '不要猜参数；保存 help 后返回分析。' >&2
  false
fi
```

检查端口：

```bash
ss -ltnp | grep -E ":${AUTOKV_PORT}[[:space:]]" || true
```

如果端口被占用：

```bash
export AUTOKV_OCCUPYING_PID='把 ss 显示的 PID 填在这里'
ps -fp "$AUTOKV_OCCUPYING_PID"
```

只有确认它是你自己此前启动的 AutoKV vLLM 服务后，才执行：

```bash
kill -TERM "$AUTOKV_OCCUPYING_PID"
```

等待端口释放：

```bash
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -Eq ":${AUTOKV_PORT}[[:space:]]"; then
    break
  fi
  sleep 1
done

ss -ltnp | grep -E ":${AUTOKV_PORT}[[:space:]]" || true
```

不要使用 `kill -9` 或 `pkill` 处理未知进程。

## 9. 定义可复用的服务启动、停止和严格探针函数

以下整段在同一个 tmux shell 中粘贴执行：

```bash
start_autokv_server() {
  local label="$1"
  local backend="$2"
  shift 2

  local log_path="$AUTOKV_DIAG_DIR/${label}.server.log"
  local pid_path="$AUTOKV_DIAG_DIR/${label}.pid"

  if ss -ltn | grep -Eq ":${AUTOKV_PORT}[[:space:]]"; then
    printf '拒绝启动：端口 %s 已被占用\n' "$AUTOKV_PORT" >&2
    return 2
  fi

  nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_LOGGING_LEVEL=DEBUG \
    "$AUTOKV_VLLM" serve \
      --model "$AUTOKV_MODEL_PATH" \
      --served-model-name "$AUTOKV_MODEL_ID" \
      --dtype bfloat16 \
      --attention-backend "$backend" \
      --max-model-len 32768 \
      --kv-cache-memory-bytes 16G \
      --seed 42 \
      --tensor-parallel-size 1 \
      --host 127.0.0.1 \
      --port "$AUTOKV_PORT" \
      "$AUTOKV_PREFIX_ARG" \
      "$@" \
      > "$log_path" 2>&1 &

  local server_pid=$!
  printf '%s\n' "$server_pid" > "$pid_path"
  printf '已启动 %s，PID=%s，日志=%s\n' "$label" "$server_pid" "$log_path"

  for _ in $(seq 1 450); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      printf '服务提前退出：%s\n' "$label" >&2
      tail -n 120 "$log_path" >&2
      return 3
    fi
    if curl -fsS "http://127.0.0.1:${AUTOKV_PORT}/health" >/dev/null; then
      printf '服务健康：%s\n' "$label"
      return 0
    fi
    sleep 2
  done

  printf '服务 900 秒内未就绪：%s\n' "$label" >&2
  tail -n 120 "$log_path" >&2
  return 4
}

stop_autokv_server() {
  local label="$1"
  local pid_path="$AUTOKV_DIAG_DIR/${label}.pid"

  if ! test -f "$pid_path"; then
    printf '没有 PID 文件：%s\n' "$pid_path" >&2
    return 2
  fi

  local server_pid
  server_pid="$(cat "$pid_path")"

  if ! printf '%s\n' "$server_pid" | grep -Eq '^[0-9]+$'; then
    printf '拒绝停止：PID 文件内容无效：%q\n' "$server_pid" >&2
    return 3
  fi
  if test "$server_pid" -le 1; then
    printf '拒绝停止：PID 必须大于 1：%q\n' "$server_pid" >&2
    return 3
  fi

  if ! kill -0 "$server_pid" 2>/dev/null; then
    printf '进程已退出：PID=%s\n' "$server_pid"
    mv -f -- "$pid_path" "$pid_path.stopped"
    return 0
  fi

  if ! ps -p "$server_pid" -o args= | grep -Fq "$AUTOKV_MODEL_PATH"; then
    printf '拒绝停止：PID %s 的命令行不含目标模型路径\n' "$server_pid" >&2
    return 3
  fi

  kill -TERM "$server_pid"

  for _ in $(seq 1 60); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      printf '服务已停止：%s\n' "$label"
      mv -f -- "$pid_path" "$pid_path.stopped"
      return 0
    fi
    sleep 1
  done

  printf '服务在 60 秒后仍存在；不要 pkill，请检查进程树：\n' >&2
  ps -ef --forest | grep -E "${server_pid}|vllm" | grep -v grep >&2 || true
  return 4
}

run_correctness_probe() {
  local label="$1"

  env \
    AUTOKV_DIAG_LABEL="$label" \
    AUTOKV_DIAG_PORT="$AUTOKV_PORT" \
    AUTOKV_DIAG_MODEL_ID="$AUTOKV_MODEL_ID" \
    PYTHONPATH="$AUTOKV_ROOT" \
    "$AUTOKV_PYTHON" - <<'PY' | tee "$AUTOKV_DIAG_DIR/${label}.probe.json"
import json
import os
import re
import sys

from autokv.client import VllmClient, wait_until_ready
from autokv.niah import NiahCase, expected_answer, fit_prompt

label = os.environ["AUTOKV_DIAG_LABEL"]
port = os.environ["AUTOKV_DIAG_PORT"]
model_id = os.environ["AUTOKV_DIAG_MODEL_ID"]
client = VllmClient(f"http://127.0.0.1:{port}", model_id, timeout=180)
wait_until_ready(client, timeout_seconds=60, interval_seconds=1)

cases = (
    NiahCase("diag-256", 256, 0.5, 42, "KV-DIAG-0256"),
    NiahCase("diag-1024", 1024, 0.5, 42, "KV-DIAG-1024"),
)

def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()

rows = []
for case in cases:
    materialized = fit_prompt(case, client.tokenize)
    response = client.complete(materialized.prompt, 96)
    choice = response["choices"][0]
    output = str(choice.get("text", ""))
    usage = response.get("usage", {})
    output_tokens = usage.get("completion_tokens")
    expected = expected_answer(case.code)
    strict_exact = normalize(output) == normalize(expected)
    at_limit = isinstance(output_tokens, int) and output_tokens >= 96
    has_replacement = "�" in output
    passed = strict_exact and not at_limit and not has_replacement
    rows.append(
        {
            "sample_id": case.sample_id,
            "prompt_tokens": materialized.token_count,
            "expected": expected,
            "output": output,
            "output_tokens": output_tokens,
            "strict_exact": strict_exact,
            "at_limit": at_limit,
            "has_replacement_character": has_replacement,
            "passed": passed,
        }
    )

payload = {
    "label": label,
    "passed": all(row["passed"] for row in rows),
    "rows": rows,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
sys.exit(0 if payload["passed"] else 10)
PY
  local probe_status="${PIPESTATUS[0]}"
  return "$probe_status"
}

run_autokv_case() {
  local label="$1"
  local backend="$2"
  shift 2

  start_autokv_server "$label" "$backend" "$@"
  local start_status="$?"
  if test "$start_status" -ne 0; then
    printf 'CASE_START_FAILED label=%s status=%s\n' \
      "$label" "$start_status" >&2
    if test -f "$AUTOKV_DIAG_DIR/${label}.pid"; then
      stop_autokv_server "$label" || true
    fi
    return 20
  fi

  run_correctness_probe "$label"
  local probe_status="$?"

  stop_autokv_server "$label"
  local stop_status="$?"
  if test "$stop_status" -ne 0; then
    printf 'CASE_STOP_FAILED label=%s status=%s\n' \
      "$label" "$stop_status" >&2
    return 30
  fi

  if test "$probe_status" -eq 0 || test "$probe_status" -eq 10; then
    return "$probe_status"
  fi

  printf 'CASE_PROBE_INFRA_FAILED label=%s status=%s\n' \
    "$label" "$probe_status" >&2
  return 21
}
```

确认函数已定义：

```bash
type start_autokv_server
type stop_autokv_server
type run_correctness_probe
type run_autokv_case
```

`run_autokv_case` 无论正确性通过与否都会尝试停止本次服务。返回码含义固定为：`0` 正确性通过，`10` 服务正常但正确性失败，`20` 启动失败，`21` 探针基础设施异常，`30` 服务未能安全停止。只有 `0` 和 `10` 可以进入正确性结论，其他值都必须进入第 18 节。

## 10. 运行最小 A/B/C 诊断矩阵

### 10.1 A：BF16 + FlashInfer

```bash
run_autokv_case \
  'A-bf16-flashinfer' \
  'FLASHINFER' \
  --kv-cache-dtype bfloat16
export AUTOKV_A_RESULT="$?"

printf 'A_RESULT=%s\n' "$AUTOKV_A_RESULT"
```

成功判据：`A_RESULT=0`，两条样本 `strict_exact=true`。

若 A 失败，说明问题不是 FP8 专属。保存日志并直接进入第 18 节，不继续 B/C。

### 10.2 B：FP8 + 动态随机 scale

该配置用于复现已知错误，预期很可能失败：

```bash
run_autokv_case \
  'B-fp8-dynamic-scale' \
  'FLASHINFER' \
  --kv-cache-dtype fp8_e4m3 \
  --calculate-kv-scales
export AUTOKV_B_RESULT="$?"

printf 'B_RESULT=%s\n' "$AUTOKV_B_RESULT"
```

已知故障被有效复现的判据是 `B_RESULT=10`。若为 `20`、`21` 或 `30`，这是运行基础设施失败，不能当成 FP8 乱码证据；进入第 18 节。不要反复运行 B。

### 10.3 C：FP8 + 不动态计算 scale

```bash
run_autokv_case \
  'C-fp8-checkpoint-or-default-scale' \
  'FLASHINFER' \
  --kv-cache-dtype fp8_e4m3
export AUTOKV_C_RESULT="$?"

printf 'C_RESULT=%s\n' "$AUTOKV_C_RESULT"
```

一次性汇总：

```bash
printf 'A=%s B=%s C=%s\n' \
  "$AUTOKV_A_RESULT" "$AUTOKV_B_RESULT" "$AUTOKV_C_RESULT" \
  | tee "$AUTOKV_DIAG_DIR/abc-result.txt"
```

解释：

| 结果 | 结论 | 下一步 |
|---|---|---|
| A=0，B=10，C=0 | 动态随机 scale 是直接触发条件 | 进入第 11 节，重复确认 C 后应用配置修复 |
| A=0，B=10，C=10 | 不能只归因于动态 scale | 进入第 12 节，隔离 compile/backend |
| A=10 | BF16 基线本身错误 | 停止，进入第 18 节 |
| A=0，B=0，C=0 | 旧故障未复现，环境或源码已变化 | 不直接跑完整实验；核对 commit、运行时和日志差异 |
| 任一结果为 20、21 或 30 | 启动、探针或停止流程异常 | 停止，进入第 18 节；不得作正确性归因 |

## 11. C 通过时：确认并应用动态 scale 修复

先用相同 C 配置再独立启动两次。连同第一次，共 3 次服务启动：

```bash
for repeat in 2 3; do
  label="C-fp8-default-scale-repeat-${repeat}"
  run_autokv_case \
    "$label" \
    'FLASHINFER' \
    --kv-cache-dtype fp8_e4m3
  result="$?"

  printf '%s=%s\n' "$label" "$result" \
    | tee -a "$AUTOKV_DIAG_DIR/c-repeat-result.txt"

  if test "$result" -ne 0; then
    printf '%s\n' '重复确认失败；停止，不修改正式配置。' >&2
    break
  fi
done
```

只有 3 次 C 都通过，才修改 quick/full profile。以下脚本会：

- 把现场真实 driver 固定进配置；
- 把 `max_tokens` 固定为 96；
- 把 `calculate_kv_scales` 固定为 false；
- 同步 `autokv/config.py` 的批准默认值；
- 不修改其他参数。

```bash
cd "$AUTOKV_ROOT"

export AUTOKV_OBSERVED_DRIVER="$(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits \
    | head -n 1 \
    | tr -d '[:space:]'
)"

printf 'OBSERVED_DRIVER=%s\n' "$AUTOKV_OBSERVED_DRIVER"

"$AUTOKV_PYTHON" - "$AUTOKV_OBSERVED_DRIVER" <<'PY'
import json
import re
import sys
from pathlib import Path

driver = sys.argv[1]
if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", driver):
    raise SystemExit(f"无效 driver 字符串：{driver!r}")

for relative in ("configs/quick.json", "configs/full.json"):
    path = Path(relative)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["hardware"]["driver"] = driver
    data["quality"]["max_tokens"] = 96
    data["calculate_kv_scales"] = False
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

path = Path("autokv/config.py")
text = path.read_text(encoding="utf-8")

text, driver_count = re.subn(
    r'^EXPECTED_DRIVER = "[^"]+"$',
    f'EXPECTED_DRIVER = "{driver}"',
    text,
    count=1,
    flags=re.MULTILINE,
)
text, token_count = re.subn(
    r'"max_tokens":\s*(?:24|64|96),',
    '"max_tokens": 96,',
    text,
    count=1,
)
text, scale_count = re.subn(
    r'"calculate_kv_scales":\s*(?:True|False),',
    '"calculate_kv_scales": False,',
    text,
    count=1,
)

if (driver_count, token_count, scale_count) != (1, 1, 1):
    raise SystemExit(
        "拒绝写入：config.py 的目标结构与预期不同，"
        f"counts={(driver_count, token_count, scale_count)}"
    )

path.write_text(text, encoding="utf-8")

test_path = Path("tests/test_config.py")
if test_path.is_file():
    test_text = test_path.read_text(encoding="utf-8")
    old = 'lambda data: data.__setitem__("calculate_kv_scales", False)'
    new = 'lambda data: data.__setitem__("calculate_kv_scales", True)'
    if old in test_text:
        test_path.write_text(test_text.replace(old, new, 1), encoding="utf-8")

print("PROFILE_FIX_APPLIED", driver)
PY
```

立即检查差异：

```bash
git diff --check
git diff -- autokv/config.py configs/quick.json configs/full.json tests/test_config.py

python3 scripts/verify.py \
  |& tee "$AUTOKV_DIAG_DIR/verify-after-scale-fix.log"
```

若测试因为其他文件仍硬编码旧 driver 或旧 max_tokens 而失败，不要删除测试或放宽检查。按失败信息逐个同步文档/测试中的冻结值，然后重新执行完整验证。

同时定位仍然陈述旧环境或旧 scale 策略的源码、测试和文档；把它们更新为实际已验证事实，不做全局盲替换：

```bash
rg -n \
  '535\.230\.02|max_tokens.?=.?24|calculate-kv-scales|目标服务器尚未验证' \
  autokv configs tests README.md RUNBOOK.zh-CN.md docs \
  || grep -RInE \
    '535\.230\.02|max_tokens.?=.?24|calculate-kv-scales|目标服务器尚未验证' \
    autokv configs tests README.md RUNBOOK.zh-CN.md docs
```

服务器没有 `rg` 时会自动使用 `grep`。逐条判断是否需要更新，不要修改历史结果中的原始环境记录，也不要从 vLLM 能力检查中删除 `--calculate-kv-scales`：修复是默认不启用该参数，不是禁止检测当前 vLLM 是否具备该能力。

确认 dry-run 中 FP8 和 Auto 命令都不再包含 `--calculate-kv-scales`：

```bash
python3 -m autokv dry-run \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  > "$AUTOKV_DIAG_DIR/dry-run-after-scale-fix.json"

"$AUTOKV_PYTHON" - "$AUTOKV_DIAG_DIR/dry-run-after-scale-fix.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)
commands = payload["server_commands"]
for name in ("fp8", "auto-4-placeholder"):
    command = commands[name]
    if "--calculate-kv-scales" in command:
        raise SystemExit(f"{name} 仍包含 --calculate-kv-scales")
print("DRY_RUN_SCALE_FLAG_REMOVED")
PY
```

提交最小配置修复：

```bash
git add -u -- autokv configs tests README.md RUNBOOK.zh-CN.md docs
git diff --cached --check
git diff --cached --name-status
git diff --cached
git commit -m "fix: disable corrupt dynamic FP8 KV scales" &&
test -z "$(git status --porcelain)" &&
git push &&
git status --short --branch
```

## 12. C 失败时：隔离 compile 和 FlashInfer

只有 A 通过、C 失败时执行本节。

### 12.1 D：关闭 torch.compile 和 CUDA Graph

```bash
run_autokv_case \
  'D-fp8-eager' \
  'FLASHINFER' \
  --kv-cache-dtype fp8_e4m3 \
  --enforce-eager
export AUTOKV_D_RESULT="$?"

printf 'D_RESULT=%s\n' "$AUTOKV_D_RESULT"
```

- D 通过：暂时以 `--enforce-eager` 作为正确性回退；不要立即将其包装成性能优化。下一步需分别关闭 CUDAGraph 和 compile，继续缩小范围。
- D 失败：继续 E。

### 12.2 E：切换该版本实际支持的 attention backend

先从 help 中查看可选值：

```bash
grep -nA 8 -B 2 -- '--attention-backend' \
  "$AUTOKV_DIAG_DIR/vllm-serve-help.txt"
```

仅选择 help 明确列出且支持 FP8 KV 的后端。例如列表中确实存在 `FLASH_ATTN` 时：

```bash
export AUTOKV_ALT_BACKEND='FLASH_ATTN'

run_autokv_case \
  'E-fp8-alternate-backend' \
  "$AUTOKV_ALT_BACKEND" \
  --kv-cache-dtype fp8_e4m3
export AUTOKV_E_RESULT="$?"

printf 'E_RESULT=%s\n' "$AUTOKV_E_RESULT"
```

不要猜测 `TRITON_ATTN`、`FLASH_ATTN` 等名称，也不要在 help 不支持时强行传入。

- E 通过：问题位于 FlashInfer 路径；保存 B/C/E 的完整日志。替代后端只是诊断结果，尚未进入项目冻结配置。
- E 失败：不要在现有虚拟环境中随意升级包。进入第 18 节，收集 vLLM 源码 diff、FlashInfer 版本和所有诊断日志。

D 或 E 即使通过，也先到第 18 节打包证据并停止，不执行后续正式 quick run。因为当前项目把 `FLASHINFER` 固定在 profile 和测试中；在没有同步修改控制器、测试及实验声明前，手工可用配置不能冒充可复现实验修复。

## 13. 修复后重新生成环境状态和新 run

只在第 11 节的配置修复已经通过 3 次确认并提交后执行。第 12 节 D/E 分支只用于定位，不能进入本节。

先确认 Git 干净：

```bash
cd "$AUTOKV_ROOT"
git status --short --branch
git rev-parse HEAD
```

如果 status 仍有修改，不进入正式 run。

按顺序重新生成环境和数据状态。下面使用 `&&`，任一阶段失败都会阻止后续阶段执行：

```bash
python3 -m autokv doctor \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/doctor-fixed.json" &&

python3 -m autokv lock-image \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/lock-fixed.json" &&

python3 -m autokv make-data \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/make-data-fixed.json" &&

python3 -m autokv dry-run \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/dry-run-fixed.json" &&

python3 -m autokv status \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/status-before-smoke.json"
```

确认 status 中：

- `git_dirty=false`；
- `run_id` 与旧的 `02db55c61f0d14bb`、`1cf1ec252f4e7e62` 均不同；
- driver、模型 revision、runtime ID 与现场一致。

如果 doctor 仍要求 `535.230.02`，说明真实运行源码或冻结配置没有完整同步；不要修改系统 driver，返回第 11 节检查源码。

## 14. 运行控制器 smoke，并执行外部严格门禁

```bash
python3 -m autokv smoke \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --port "$AUTOKV_PORT" \
  --json \
  | tee "$AUTOKV_DIAG_DIR/controller-smoke-fixed.json"
```

当前控制器可能仍把“相同乱码”视为 deterministic，因此必须执行下面的独立严格检查：

```bash
cd "$AUTOKV_ROOT"

PYTHONPATH="$AUTOKV_ROOT" "$AUTOKV_PYTHON" - <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path.cwd()
status = json.loads(
    __import__("subprocess").check_output(
        [
            sys.executable,
            "-m",
            "autokv",
            "status",
            "--project-root",
            str(root),
            "--profile",
            "quick",
            "--json",
        ],
        text=True,
    )
)
run_id = status["run_id"]
if not run_id:
    raise SystemExit("status 没有 run_id")

profile = json.loads((root / "configs/quick.json").read_text(encoding="utf-8"))
max_tokens = int(profile["quality"]["max_tokens"])
smoke = json.loads(
    (root / "runs" / run_id / "smoke" / "smoke.json").read_text(encoding="utf-8")
)

def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()

checks = []
for key in ("first", "second"):
    relative = Path(smoke[key]["path"])
    lines = (root / relative).read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise SystemExit(f"{key} 应只有一条记录，实际 {len(lines)}")
    row = json.loads(lines[0])
    output = str(row.get("output_text", ""))
    expected = str(row.get("expected", ""))
    output_tokens = row.get("output_tokens")
    strict_exact = normalize(output) == normalize(expected)
    passed = (
        strict_exact
        and "�" not in output
        and isinstance(output_tokens, int)
        and output_tokens < max_tokens
        and row.get("exact_match") == 1.0
    )
    checks.append(
        {
            "key": key,
            "strict_exact": strict_exact,
            "output_tokens": output_tokens,
            "max_tokens": max_tokens,
            "answer_nll": row.get("answer_nll"),
            "passed": passed,
        }
    )

payload = {"run_id": run_id, "passed": all(x["passed"] for x in checks), "checks": checks}
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload["passed"]:
    raise SystemExit("STRICT_SMOKE_GATE_FAILED")
print("STRICT_SMOKE_GATE_OK")
PY
```

只有出现 `STRICT_SMOKE_GATE_OK` 才进入 probe。

## 15. 分阶段运行 quick，不使用一键 `run`

不要立即执行 `python3 -m autokv run`。逐阶段执行，才能在错误进入 benchmark 前停止。

### 15.1 Probe

```bash
python3 -m autokv probe \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --port "$AUTOKV_PORT" \
  --json \
  | tee "$AUTOKV_DIAG_DIR/probe-fixed.json"
```

执行退化输出检查：

```bash
cd "$AUTOKV_ROOT"

PYTHONPATH="$AUTOKV_ROOT" "$AUTOKV_PYTHON" - <<'PY'
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
status = json.loads(
    subprocess.check_output(
        [sys.executable, "-m", "autokv", "status", "--project-root", str(root), "--profile", "quick", "--json"],
        text=True,
    )
)
run_id = status["run_id"]
index = json.loads((root / "runs" / run_id / "probe" / "index.json").read_text(encoding="utf-8"))
max_tokens = json.loads((root / "configs/quick.json").read_text(encoding="utf-8"))["quality"]["max_tokens"]

def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()

def summarize(name: str):
    relative = Path(index["artifacts"][name]["path"])
    rows = [json.loads(line) for line in (root / relative).read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = [normalize(str(row.get("output_text", ""))) for row in rows]
    nlls = [float(row["answer_nll"]) for row in rows if isinstance(row.get("answer_nll"), (int, float))]
    summary = {
        "name": name,
        "samples": len(rows),
        "exact_count": sum(float(row.get("exact_match", 0.0)) == 1.0 for row in rows),
        "unique_outputs": len(set(outputs)),
        "at_token_limit": sum(row.get("output_tokens") == max_tokens for row in rows),
        "replacement_character": sum("�" in str(row.get("output_text", "")) for row in rows),
        "finite_nll": len(nlls) == len(rows) and all(math.isfinite(value) for value in nlls),
        "mean_nll": statistics.fmean(nlls) if nlls else None,
    }
    summary["non_degenerate"] = (
        summary["samples"] > 0
        and summary["unique_outputs"] >= max(2, summary["samples"] // 2)
        and summary["at_token_limit"] < summary["samples"]
        and summary["replacement_character"] == 0
        and summary["finite_nll"]
    )
    return summary

summaries = [summarize("fp8"), summarize("auto-4")]
print(json.dumps({"run_id": run_id, "summaries": summaries}, ensure_ascii=False, indent=2))
if not all(item["non_degenerate"] for item in summaries):
    raise SystemExit("PROBE_DEGENERATE_OUTPUT_GATE_FAILED")
print("PROBE_DEGENERATE_OUTPUT_GATE_OK")
PY
```

如果失败，停止；不要 select/evaluate。执行第 18 节。

### 15.2 Select 与 Evaluate

```bash
python3 -m autokv select \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/select-fixed.json" &&

python3 -m autokv evaluate \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --port "$AUTOKV_PORT" \
  --json \
  | tee "$AUTOKV_DIAG_DIR/evaluate-fixed.json"
```

质量验收沿用该次运行源码已有的 `exact_match`、`quality_score` 和 `answer_nll` 语义。本手册此前草拟的额外质量指标与门禁已撤销：不得在复验阶段另建一个不同的 EM 变体，也不得用新指标追溯决定本次运行是否通过。修复后运行的实际质量结果与解释统一见 [v1.0 统一项目事实](docs/v1.0/FACTS.zh-CN.md)。

## 16. Benchmark、报告和完整性检查

```bash
python3 -m autokv benchmark \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --port "$AUTOKV_PORT" \
  --json \
  | tee "$AUTOKV_DIAG_DIR/benchmark-fixed.json" &&

python3 -m autokv report \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/report-fixed.json" &&

python3 -m autokv status \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/status-complete.json"
```

取得 run ID：

```bash
export AUTOKV_RUN_ID="$(
  python3 -m autokv status \
    --project-root "$AUTOKV_ROOT" \
    --profile quick \
    --json \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])'
)"

printf 'RUN_ID=%s\n' "$AUTOKV_RUN_ID"

if printf '%s\n' "$AUTOKV_RUN_ID" | grep -Eq '^[0-9a-f]{16}$'; then
  printf '%s\n' 'RUN_ID_FORMAT_OK'
else
  printf '停止：无效 RUN_ID=%q\n' "$AUTOKV_RUN_ID" >&2
  unset AUTOKV_RUN_ID
fi
```

只有出现 `RUN_ID_FORMAT_OK` 才继续。变量被清空时不要执行后续打包命令。

检查核心文件：

```bash
test -f "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/run-manifest.json"
test -f "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/probe/index.json"
test -f "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/selection.json"
test -f "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/quality/index.json"
test -f "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/perf/index.json"
test -f "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/report/REPORT.zh-CN.md"
test -f "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/completed-manifest.json"

python3 -m json.tool \
  "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/run-manifest.json" \
  | sed -n '1,160p'

python3 -m json.tool \
  "$AUTOKV_ROOT/runs/$AUTOKV_RUN_ID/completed-manifest.json" \
  | sed -n '1,120p'
```

manifest 必须显示干净 Git 身份和本次实际运行时。若 `git_dirty=true`，本次结果不能作为最终可复现结果。

注意：当前 quick benchmark 只有 1 次重复，因此仍是预跑，不是最终面试性能数字。正式性能完善见第 20 节。

## 17. 打包并从服务器取回结果

先生成项目自己的脱敏诊断包：

```bash
python3 -m autokv diagnose \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/diagnose-final.json"
```

打包本次 run 和 bugfix 诊断目录，不包含模型缓存：

```bash
if printf '%s\n' "${AUTOKV_RUN_ID:-}" | grep -Eq '^[0-9a-f]{16}$'; then
  export AUTOKV_EXPORT_DIR="$AUTOKV_BACKUP_ROOT"
  export AUTOKV_EXPORT_NAME="autokv-fp8-fixed-$AUTOKV_RUN_ID.tar.gz"
  export AUTOKV_EXPORT_PATH="$AUTOKV_EXPORT_DIR/$AUTOKV_EXPORT_NAME"

  tar \
    -C "$AUTOKV_ROOT" \
    -czf "$AUTOKV_EXPORT_PATH" \
    "runs/$AUTOKV_RUN_ID" \
    "runs/_bugfix/$AUTOKV_FIX_TAG" \
  && (
    cd "$AUTOKV_EXPORT_DIR" || exit 1
    sha256sum "$AUTOKV_EXPORT_NAME" \
      | tee "$AUTOKV_EXPORT_NAME.sha256"
  ) \
  && ls -lh "$AUTOKV_EXPORT_PATH" "$AUTOKV_EXPORT_PATH.sha256"
else
  printf '%s\n' '拒绝打包：AUTOKV_RUN_ID 为空或格式错误' >&2
fi
```

退出 SSH 前再次确认 Git：

```bash
cd "$AUTOKV_ROOT"
git status --short --branch
git log -3 --oneline --decorate
git push
```

按 `Ctrl-b`、再按 `d` 可从 tmux 脱离。随后退出 SSH：

```bash
exit
```

回到 Linux 客户端后下载。把服务器导出的真实路径填入第二行：

```bash
export AUTOKV_SSH_TARGET='LOGIN_NAME@SERVER_ADDRESS'
export AUTOKV_REMOTE_EXPORT='/mnt_d/autokv-backups/autokv-fp8-fixed-RUN_ID.tar.gz'

scp "$AUTOKV_SSH_TARGET:$AUTOKV_REMOTE_EXPORT" . &&
scp "$AUTOKV_SSH_TARGET:$AUTOKV_REMOTE_EXPORT.sha256" . &&

sha256sum -c "$(basename "$AUTOKV_REMOTE_EXPORT").sha256"
```

如果使用非默认 SSH 端口，给两条 `scp` 增加大写 `-P PORT`。

## 18. 失败时收集诊断并停止

无论在哪个阶段失败，都执行：

```bash
cd "$AUTOKV_ROOT"

git status --short --branch \
  | tee "$AUTOKV_DIAG_DIR/failure-git-status.txt"

nvidia-smi \
  | tee "$AUTOKV_DIAG_DIR/failure-nvidia-smi.txt"

nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits \
  | tee "$AUTOKV_DIAG_DIR/failure-gpu-processes.txt"

ss -ltnp \
  | tee "$AUTOKV_DIAG_DIR/failure-listeners.txt"

python3 -m autokv status \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/failure-status.json" || true

python3 -m autokv diagnose \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json \
  | tee "$AUTOKV_DIAG_DIR/failure-diagnose.json" || true

find "$AUTOKV_DIAG_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

若本手册启动的服务还活着，只按 PID 文件停止：

```bash
for pid_file in "$AUTOKV_DIAG_DIR"/*.pid; do
  test -e "$pid_file" || continue
  label="$(basename "$pid_file" .pid)"
  stop_autokv_server "$label" || true
done
```

不要为了“清干净”而删除任何 run、cache 或备份。带回以下内容即可继续分析：

- `environment.txt`；
- `python-runtime.json`；
- `vllm-serve-help.txt`；
- A/B/C/D/E 的 `*.server.log` 和 `*.probe.json`；
- vLLM 源码 commit、status 和 diff；
- `failure-status.json`；
- 项目生成的脱敏 diagnose 包。

## 19. 永久代码修复清单

第 14～15 节的外部脚本是现场保护门禁。确认根因后，应把相同规则正式写进项目代码并增加回归测试：

1. `run_smoke` 增加 BF16 正确性基线；
2. 两次 FP8 必须既相同又正确；
3. 严格比较归一化完整答案，不再用“期望答案是输出子串”代替 Exact Match；
4. 输出达到 `max_tokens`、含 `�`、NLL 非有限值时 smoke 必须失败；
5. probe 在发现全量 token-limit、单一循环输出时立即停止；
6. Auto-4 在 selection 后增加独立正确性 gate；
7. benchmark 必须依赖最终 quality gate，而不只是 smoke 和 selection；
8. 启动时验证 `max_tokens >= tokenizer(expected_answer) + 8`；
9. 生成与 NLL 评分使用同一个 prompt 上下文；
10. 为“两个相同乱码仍然失败”增加单元测试。

每次只提交一个可验证修改，完整运行 `python3 scripts/verify.py`。

## 20. Bug 修复后的项目完善顺序

修复 run 完整通过后，再按以下顺序完善，不在本次现场排障中同时扩张范围。

### 20.1 Scale 标定

- 使用 llm-compressor 和代表性数据集生成 checkpoint KV scales；
- 从 128～512 条、2048-token 样本开始；
- 补充少量 8192-token 长上下文样本；
- 记录每层 scale 的来源、数值范围和 finite 状态；
- 比较默认 1.0、随机 warmup、数据集标定三种方案。

### 20.2 选层可信度

- probe 与 final 使用不同样本和 seed；
- full 模式覆盖 32 层和 3 个 probe seed；
- 使用配对样本的 ΔNLL 和 bootstrap 区间；
- 比较 `k=0/2/4/8`；
- 得分差小于统计噪声时报告“层排名不确定”。

### 20.3 性能实验

- Prefix cache 默认关闭，单独建立 cache 实验；
- 每个场景至少重复 3 次，正式结果最好 5 次；
- 同时报告固定并发与饱和并发；
- 增加有限 request rate；
- 配置顺序随机化或交错运行；
- 报告中位数和置信区间，不跨异构长度平均延迟。

### 20.4 泛化和面试材料

- 增加一个第二模型，不扩展成大规模 benchmark；
- 增加一个自然文本长上下文或 perplexity 任务；
- README 只保留问题、方法、关键结果、失败诊断和复现入口；
- 展示同一干净 run 的质量—容量—吞吐 Pareto 图；
- 面试中把本次错误门禁和根因隔离作为核心工程案例。

## 21. 官方依据

- [vLLM Quantized KV Cache：三种 scale 标定方式与 skip-layers](https://github.com/vllm-project/vllm/blob/c18d29d36a547cf0bda7c2e8b8a1cf2308c6c5de/docs/features/quantization/quantized_kvcache.md)
- [vLLM 对 `--enforce-eager` 和 compile/CUDA Graph 的说明](https://github.com/vllm-project/vllm/blob/main/docs/design/debug_vllm_compile.md)
- [llm-compressor KV Cache 数据集标定示例](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_kv_cache/README.md)
- [相似的动态 KV scale 导致乱码问题；仅作为诊断参考，不视为本项目根因证明](https://github.com/vllm-project/vllm/issues/37554)

## 22. 最终现场记录模板

操作结束前填写：

```text
日期：
服务器：
GPU / driver：
AutoKV commit：
AutoKV branch：
AutoKV git_dirty：
vLLM version / commit：
FlashInfer version：
PyTorch / CUDA runtime：

A BF16 + FlashInfer：通过 / 失败
B FP8 + dynamic scale：通过 / 失败
C FP8 + checkpoint/default scale：通过 / 失败
D FP8 + eager：未运行 / 通过 / 失败
E FP8 + alternate backend：未运行 / 通过 / 失败

确认的直接触发条件：
采用的临时修复：
3 次独立 FP8 启动：通过 / 失败
严格 controller smoke：通过 / 失败
probe 非退化门禁：通过 / 失败
final quality 门禁：通过 / 失败
benchmark 是否运行：

新 run ID：
completed-manifest：存在 / 不存在
导出包路径：
导出包 SHA-256：
未解决问题：
```
