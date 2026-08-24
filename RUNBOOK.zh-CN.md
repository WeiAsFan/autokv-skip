# AutoKV-Skip A6000 独立运行手册

版本日期：2026-08-25

目标：在单张 NVIDIA RTX A6000 48 GiB、驱动严格为 `535.230.02` 的 Linux 服务器上，从零完成 AutoKV-Skip quick 实验；SSH 中断后可恢复；全程不需要修改驱动，也不依赖宿主 CUDA toolkit。

最重要的规则：**不要升级驱动，不要降级驱动，不要重装驱动，不要安装 CUDA runfile。** `nvidia-smi` 显示的 CUDA 12.2 是驱动能支持的 CUDA API 上限提示，不要求宿主存在 `nvcc`。实验使用 vLLM Docker 镜像内的 CUDA 和预装 forward-compat libraries，并强制设置 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`。

## 先读：作用域、时长和退出码

- quick 首次下载：20–90 分钟，取决于网络；GPU 实验保守估计 5–12 小时。
- full：12–24 小时，只在 quick 完整后选择性执行。
- 宿主建议：Linux x86_64、Python 3.10+、至少 64 GiB RAM、项目所在文件系统至少 80 GiB 可用空间。
- 所有命令默认从仓库根目录执行。若路径不同，只需让 `AUTOKV_ROOT` 指向实际根目录。
- `--json` 的标准输出是一条 JSON；诊断文字写到标准错误。

| 退出码 | 含义 | 自动行为 |
|---:|---|---|
| 0 | 成功 | 可以进入下一阶段 |
| 退出码 2 | 输入或不可变 gate 不满足 | 停止；核对目标服务器，不修改驱动 |
| 退出码 3 | Docker、文件系统或外部命令失败 | 停止；收集诊断，修复对应外部条件 |
| 退出码 4 | vLLM HTTP、健康检查或超时 | 保存当前精确配置；同命令可恢复 |
| 退出码 5 | 前置产物不完整或 smoke 不确定 | 执行错误消息给出的前置命令，不进入 probe |

每个阶段都按“命令 → 预期时间 → 成功判据 → 失败分支”组织。不要跳阶段。

---

## 阶段 0：打印清单并确认拿到完整仓库

### 命令

```bash
pwd -P
test -f pyproject.toml
test -f RUNBOOK.zh-CN.md
test -f configs/quick.json
test -f autokv/cli.py
python3 - <<'PY'
items = [
    "0 完整仓库",
    "1 Linux/Python/项目变量",
    "2 只读宿主检查",
    "3 Docker 与 NVIDIA runtime",
    "4 可选 HF_TOKEN",
    "5 doctor",
    "6 不可变镜像/模型锁",
    "7 磁盘与模型缓存",
    "8 数据与 dry-run",
    "9 双启动 smoke",
    "10 tmux quick",
    "11 查看/重连/恢复",
    "12 验收与导出",
    "13 可选 full",
]
print("\n".join(f"[ ] {item}" for item in items))
PY
```

若使用 Git 复制了仓库，再记录版本；若是压缩包复制，`git` 命令失败不影响实验：

```bash
git status --short --branch 2>/dev/null || true
git rev-parse HEAD 2>/dev/null || true
```

### 预期时间

1 分钟。

### 成功判据

四个 `test -f` 均无输出且退出码为 0；清单显示阶段 0–13。

### 失败分支

1. 缺少任一文件：不要只复制单个 Python 文件；重新复制整个 `autokv-skip` 目录。
2. 文件权限不可读：执行 `ls -la` 查明所有者；只修正本项目目录权限，不对系统目录做递归权限修改。
3. Git 显示本地改动：可以继续，但先把 `git diff` 保存；正式报告会锁 profile/data/image/model，不会自动覆盖源码。

---

## 阶段 1：识别 Linux 发行版并设置项目局部变量

### 命令

```bash
uname -a
cat /etc/os-release
uname -m
python3 --version
python3 - <<'PY'
import platform, sys
assert sys.version_info >= (3, 10), sys.version
assert platform.system() == "Linux", platform.system()
assert platform.machine() in {"x86_64", "AMD64"}, platform.machine()
print("AUTOKV_HOST_BASE_OK", sys.version.split()[0], platform.machine())
PY
```

进入仓库根目录后设置变量；以后每次新 SSH shell 都重新执行这四行：

```bash
AUTOKV_ROOT="$(pwd -P)"
export AUTOKV_ROOT
export PYTHONPATH="$AUTOKV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$AUTOKV_ROOT"
printf 'AUTOKV_ROOT=%s\n' "$AUTOKV_ROOT"
```

### 预期时间

1–2 分钟。

### 成功判据

出现 `AUTOKV_HOST_BASE_OK`；架构为 `x86_64`；Python 至少 3.10；`AUTOKV_ROOT` 末尾是项目目录。

### 失败分支

1. Python 低于 3.10：安装一个并行的 Python 3.10+ 用户环境，或让管理员提供；不要因此修改 NVIDIA 驱动或宿主 CUDA。
2. 不是 Linux/x86_64：该机器只能执行 dry-run 和报告，不能执行真实 GPU 阶段。
3. 路径含换行或不可写：把完整项目复制到一个普通、可写、至少 80 GiB 可用的路径；不要移动 Hugging Face cache 的已有文件。

---

## 阶段 2：只读宿主检查

这一阶段没有安装命令。

### 命令

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
free -h
df -h "$AUTOKV_ROOT"
df -Pk "$AUTOKV_ROOT"
uname -r
```

执行机器可判定检查：

```bash
python3 - <<'PY'
import subprocess
line = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"],
    check=True, text=True, capture_output=True,
).stdout.strip()
rows = [row for row in line.splitlines() if row.strip()]
assert len(rows) == 1, f"expected one GPU, got {len(rows)}"
name, driver, memory, capability = [field.strip() for field in rows[0].split(",")]
assert name == "NVIDIA RTX A6000", name
assert driver == "535.230.02", driver
assert int(memory) >= 48000, memory
assert capability == "8.6", capability
print("AUTOKV_GPU_GATE_OK", name, driver, memory, capability)
PY
```

检查磁盘至少 80 GiB：

```bash
python3 - <<'PY'
import os
root = os.environ["AUTOKV_ROOT"]
free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
print(f"free_gib={free / 2**30:.1f}")
assert free >= 80 * 2**30, "less than 80 GiB free"
PY
```

### 预期时间

2 分钟。

### 成功判据

出现 `AUTOKV_GPU_GATE_OK NVIDIA RTX A6000 535.230.02 ... 8.6`；只有一张 GPU；显存不低于 48000 MiB；磁盘检查通过。GPU 进程查询空输出最理想。

### 失败分支

1. 驱动不是精确的 `535.230.02`：确认 SSH 主机是否正确。**不要升级驱动**，也不要为迁就另一驱动编辑 profile。
2. GPU 型号、数量或 SM 不符：停止；这会改变容量和性能结论。
3. 有未知 GPU 进程：先联系进程所有者并等待资源释放；项目不会结束外部进程。
4. 磁盘不足：清理项目外由你确认可删除的文件，或把整个项目复制到更大文件系统。不要执行宽泛的 Docker 清理，也不要删除现有模型 cache。
5. RAM 少于 64 GiB：仍可能运行，但模型下载/加载和 benchmark 更易失败；优先换到满足条件的目标节点。

---

## 阶段 3：Docker 与 NVIDIA Container Toolkit gate

先只检查，不安装。

### 命令

```bash
command -v docker
docker version
docker info --format '{{json .Runtimes}}'
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 \
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits
```

### 预期时间

已有镜像时 1 分钟；首次拉取 CUDA base image 时 2–15 分钟。

### 成功判据

Docker server 可访问；最后一条容器命令显示 `NVIDIA RTX A6000, 535.230.02, 8.6`。这证明 Docker 的 GPU 注入可用，但还不证明 vLLM 镜像兼容；后者由 doctor 验证。

### 失败分支

1. `permission denied` 但 `sudo docker version` 成功：让管理员把当前用户加入受信任的 docker 访问方式，重新登录后重试。docker 组等价于高权限，不要在多人不可信主机随意授权。
2. Docker daemon 未启动但已安装：`sudo systemctl start docker`，然后重试。不要改 Docker data-root，除非管理员明确知道现有镜像位置。
3. Docker 未安装：只执行与你的 `/etc/os-release` 对应的“Docker 修复附录”，先审查模拟安装清单。
4. Docker 可用但 `--gpus all` 失败：只执行“Toolkit 修复附录”。不要触碰 driver、DKMS、内核模块或 Secure Boot。
5. base image 拉取超时：检查 DNS、代理和 Docker Hub 网络；不要用另一 CUDA 版本掩盖网络故障。

### Docker 修复附录 A：Ubuntu/Debian

这部分会安装 Docker，但明确拒绝任何含 driver/DKMS/kernel 的安装计划。若服务器由管理员托管，优先把下面的模拟清单交给管理员审核。

```bash
. /etc/os-release
case "$ID" in ubuntu|debian) ;; *) printf 'wrong distro: %s\n' "$ID" >&2; exit 2;; esac
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL "https://download.docker.com/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
ARCH="$(dpkg --print-architecture)"
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"
printf '%s\n' \
  "Types: deb" \
  "URIs: https://download.docker.com/linux/$ID" \
  "Suites: $CODENAME" \
  "Components: stable" \
  "Architectures: $ARCH" \
  "Signed-By: /etc/apt/keyrings/docker.asc" | \
  sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null
sudo apt-get update
sudo apt-get -s install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin | tee /tmp/autokv-docker-plan.txt
if grep -Eqi 'nvidia-(driver|dkms|kernel)|cuda-drivers' /tmp/autokv-docker-plan.txt; then
  printf '%s\n' 'REFUSE: simulated Docker plan touches the GPU driver' >&2
  exit 2
fi
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
docker version || sudo docker version
```

来源：[Docker Engine Ubuntu 安装文档](https://docs.docker.com/engine/install/ubuntu/)。Debian 的 repository URL 由 `$ID` 自动选择；若发行版是派生版，应由管理员按官方支持版本调整，而不是猜测 codename。

### Docker 修复附录 B：RHEL/Rocky/Alma/CentOS

```bash
. /etc/os-release
case "$ID" in rhel|rocky|almalinux|centos) ;; *) printf 'wrong distro: %s\n' "$ID" >&2; exit 2;; esac
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install --assumeno docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>&1 | tee /tmp/autokv-docker-plan.txt || true
if grep -Eqi 'nvidia-(driver|dkms|kernel)|cuda-drivers' /tmp/autokv-docker-plan.txt; then
  printf '%s\n' 'REFUSE: simulated Docker plan touches the GPU driver' >&2
  exit 2
fi
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
docker version || sudo docker version
```

若组织镜像仓库替换了 CentOS URL，不要自行混用发行版仓库；让管理员提供内部镜像配置。

### NVIDIA Container Toolkit 修复附录 A：Ubuntu/Debian

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get -s install --no-install-recommends nvidia-container-toolkit | tee /tmp/autokv-toolkit-plan.txt
if grep -Eqi 'nvidia-(driver|dkms|kernel)|cuda-drivers' /tmp/autokv-toolkit-plan.txt; then
  printf '%s\n' 'REFUSE: simulated toolkit plan touches the GPU driver' >&2
  exit 2
fi
sudo apt-get install -y --no-install-recommends nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### NVIDIA Container Toolkit 修复附录 B：RHEL/Rocky/Alma/CentOS

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
sudo dnf install --assumeno nvidia-container-toolkit 2>&1 | tee /tmp/autokv-toolkit-plan.txt || true
if grep -Eqi 'nvidia-(driver|dkms|kernel)|cuda-drivers' /tmp/autokv-toolkit-plan.txt; then
  printf '%s\n' 'REFUSE: simulated toolkit plan touches the GPU driver' >&2
  exit 2
fi
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Toolkit 来源：[NVIDIA Container Toolkit 安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。`nvidia-ctk` 只配置 Docker runtime；完成后回到阶段 3 开头重新运行全部 gate。

---

## 阶段 4：可选 Hugging Face token，不回显、不落盘

Mistral-7B-Instruct-v0.3 当前是公开模型，通常不需要 token。若服务器网络或 HF 策略要求 token，才执行第一组；否则执行 `unset HF_TOKEN`。

### 命令

有 token：

```bash
read -r -s -p 'HF_TOKEN: ' HF_TOKEN
printf '\n'
export HF_TOKEN
python3 - <<'PY'
import os
value = os.environ.get("HF_TOKEN", "")
assert value, "HF_TOKEN is empty"
print("HF_TOKEN_PRESENT", len(value))
PY
```

无 token：

```bash
unset HF_TOKEN
printf '%s\n' 'HF_TOKEN_UNSET_PUBLIC_MODEL_PATH'
```

### 预期时间

1 分钟。

### 成功判据

只显示 token 长度或 `HF_TOKEN_UNSET_PUBLIC_MODEL_PATH`，终端历史和项目文件中没有 token 值。

### 失败分支

1. 不慎回显 token：立即在 Hugging Face 撤销该 token，创建只读替代 token，再开新 shell。
2. 不要把 token 写入 `.env`、命令参数、README 或诊断说明。控制器只把环境变量名传给容器并在诊断包中脱敏。
3. HF 401/403：重新输入有效只读 token，然后重跑原命令；这不是镜像兼容问题，不允许切换 vLLM 版本。

---

## 阶段 5：运行 doctor，验证 R535 + 容器 CUDA + vLLM feature gate

doctor 是只读宿主检查加容器探测。它会拉取固定 tag、运行 CUDA/FlashInfer/CLI 探针、解析模型 commit，然后写 `doctor.json` 和 `lock.json`。它不会安装包或改驱动。

### 命令

```bash
cd "$AUTOKV_ROOT"
mkdir -p operator-logs
set -o pipefail
python3 -m autokv doctor \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json | tee operator-logs/doctor.json.stdout
printf 'doctor_exit=%s\n' "$?"
```

### 预期时间

已缓存镜像 2–10 分钟；首次拉取 20–90 分钟。拉取长时间无输出时可在另一个 SSH 查看 `docker images`，不要中断后改版本。

### 成功判据

退出码 0；JSON 中 `ok=true`、`driver="535.230.02"`、`image_ref` 含 `@sha256:`、`model_revision` 为 40 位十六进制。以下文件存在：

```bash
python3 -m json.tool runs/_environment/doctor.json | sed -n '1,120p'
python3 -m json.tool runs/_environment/lock.json | sed -n '1,160p'
```

`doctor.json` 的 gates 全为 true；事件中 CUDA probe 看到唯一 A6000、SM 8.6，并记录 vLLM、torch、容器 CUDA、FlashInfer 版本。

### 失败分支

1. `host does not match`：回到阶段 2；不要修改 profile 或驱动。
2. Docker pull 的 network/auth/disk 分类：修复网络、登录或磁盘，然后运行同一 doctor；这些错误不会触发版本回退。
3. 主镜像 CUDA/PTX/feature probe 失败：程序只在允许的兼容类别下自动尝试 `v0.19.1`；不要手工强制回退。
4. 两个镜像都失败：执行诊断命令，停止 GPU 流程：

```bash
python3 -m autokv diagnose --project-root "$AUTOKV_ROOT" --profile quick --json
```

5. 不要通过挂载宿主 `/usr/local/cuda`、添加高权限容器或源码编译规避 gate。

---

## 阶段 6：确认不可变镜像与模型锁

doctor 成功时已经创建 lock；`lock-image` 只在当前 host 仍满足精确驱动 gate 时复用它。

### 命令

```bash
python3 -m autokv lock-image \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json | tee operator-logs/lock-image.json.stdout
```

独立核验 digest 和模型 revision：

```bash
python3 - <<'PY'
import json, os, re
from pathlib import Path
root = Path(os.environ["AUTOKV_ROOT"])
lock = json.loads((root / "runs/_environment/lock.json").read_text())
assert re.search(r"@sha256:[0-9a-f]{64}$", lock["image_ref"])
assert lock["image_ref"].endswith("@" + lock["image_digest"])
assert re.fullmatch(r"[0-9a-f]{40}", lock["model_revision"])
assert lock["host"]["driver"] == "535.230.02"
assert lock["compatibility_env"] == "VLLM_ENABLE_CUDA_COMPATIBILITY=1"
print("AUTOKV_LOCK_OK", lock["image_ref"], lock["model_revision"])
PY
IMAGE_REF="$(python3 -c 'import json; print(json.load(open("runs/_environment/lock.json"))["image_ref"])')"
docker image inspect "$IMAGE_REF" --format '{{json .RepoDigests}}'
```

### 预期时间

1–2 分钟。

### 成功判据

出现 `AUTOKV_LOCK_OK`；`reused=true`；Docker inspect 可找到同一 RepoDigest。正式运行以后只用 `image_ref`，不再解析可变 tag。

### 失败分支

1. lock hash 不一致：不要手改 JSON；保存现有文件到诊断包，删除/替换需由你明确决定。本项目默认停止而不是覆盖可疑锁。
2. Docker 找不到 digest：重新运行 doctor 拉取并重新探测，不要把 tag 直接写成结果依据。
3. 驱动在阶段 2 后发生变化：`lock-image` 会退出 2；停止并核对管理员变更。

---

## 阶段 7：检查空间并预下载锁定 revision

模型下载使用 lock 中的同一个不可变镜像和 40 位 revision，只写项目局部 `.cache/huggingface`。不需要宿主 `transformers`、PyTorch 或 CUDA。

### 命令

```bash
cd "$AUTOKV_ROOT"
python3 - <<'PY'
import os
stats = os.statvfs(os.environ["AUTOKV_ROOT"])
free = stats.f_bavail * stats.f_frsize
print(f"free_before_download_gib={free / 2**30:.1f}")
assert free >= 80 * 2**30, "need at least 80 GiB before first download"
PY
mkdir -p .cache/huggingface
IMAGE_REF="$(python3 -c 'import json; print(json.load(open("runs/_environment/lock.json"))["image_ref"])')"
MODEL_REV="$(python3 -c 'import json; print(json.load(open("runs/_environment/lock.json"))["model_revision"])')"
docker run --rm \
  --label io.autokv.project=autokv-skip \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN \
  -e MODEL_REV="$MODEL_REV" \
  -v "$AUTOKV_ROOT/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint python \
  "$IMAGE_REF" \
  -c 'import os; from huggingface_hub import snapshot_download; p=snapshot_download("mistralai/Mistral-7B-Instruct-v0.3", revision=os.environ["MODEL_REV"]); print("MODEL_CACHE_OK", p)'
du -sh .cache/huggingface
df -h "$AUTOKV_ROOT"
```

### 预期时间

已缓存 1–3 分钟；首次下载 15–90 分钟。

### 成功判据

出现 `MODEL_CACHE_OK`；命令退出 0；cache 通常占十几 GiB；下载后仍有足够空间保存镜像和实验结果。

### 失败分支

1. HF 401/403：回到阶段 4 设置只读 token，重跑；snapshot download 会复用已完成文件。
2. 网络中断：重跑同一命令，禁止换 revision。
3. `No space left on device`：不要删除 `.cache/huggingface` 中部分文件制造损坏快照；扩容或把整个项目迁移到更大文件系统后从阶段 1 重来。
4. 这一下载命令不使用 GPU；若它失败，与 FP8/FlashInfer 无关。

---

## 阶段 8：生成数据、检查 hash、执行两档 dry-run

### 命令

```bash
python3 -m autokv make-data \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json | tee operator-logs/make-data-quick.json.stdout
python3 -m autokv dry-run \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json | tee operator-logs/dry-run-quick.json.stdout
python3 -m autokv dry-run \
  --project-root "$AUTOKV_ROOT" \
  --profile full \
  --json | tee operator-logs/dry-run-full.json.stdout
sha256sum configs/quick.json data/niah/quick-probe.jsonl data/niah/quick-final.jsonl data/niah/quick-manifest.json
python3 -m json.tool data/niah/quick-manifest.json
```

可选地再次做发布级本地验证：

```bash
python3 scripts/verify.py
```

### 预期时间

1–3 分钟；不访问 Docker、HTTP 或 GPU。

### 成功判据

- quick：`core_probe_configurations=18`、`probe_samples_per_configuration=6`、`quality_configurations=11`、`quality_samples_per_configuration=18`、`benchmark_scenarios=6`、`executed=false`；
- full：`core_probe_configurations=34`、`benchmark_scenarios=18`、`executed=false`；
- manifest 中 quick probe=6、final=18，`dataset_hash` 为 64 位十六进制；
- server command example 含固定 backend、16G、FP8、skip layers、compatibility env，且不含高权限模式或宿主 CUDA mount。

### 失败分支

1. 配置数不符：停止；这意味着 profile 或代码与批准版本不同。
2. 第二次 `make-data` hash 变化：停止并保存两份差异；确定性数据不应变化。
3. `scripts/verify.py` 失败：不要上 GPU；按照首个失败测试修复代码或恢复完整仓库。

---

## 阶段 9：10 分钟级双启动 FP8 smoke

smoke 会从干净 engine 独立启动相同 FP8 配置两次，发送同一 1024-token NIAH 请求，并比较输出文本、output token 数和日志解析出的 KV token capacity。任一不一致都禁止层排序。

### 命令

先确认 GPU 空闲：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
docker ps --filter label=io.autokv.project=autokv-skip
```

执行 smoke：

```bash
set -o pipefail
python3 -m autokv smoke \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --port 8000 \
  --json | tee operator-logs/smoke-quick.json.stdout
```

检查证据：

```bash
python3 -m autokv status --project-root "$AUTOKV_ROOT" --profile quick --json
find runs -path '*/smoke/smoke.json' -o -path '*/smoke-*.server.log' -print
grep -RHiE 'FLASHINFER|FP8|GPU KV cache size|Maximum concurrency' runs/*/smoke-*/*.server.log
```

### 预期时间

10–30 分钟；第一次 engine 可能要编译 kernel/cache。

### 成功判据

退出码 0；`smoke.json` 中 `complete=true`、`deterministic=true`，三个 comparison 字段都为 true，两次 capacity 完全相等；两次日志都确认 FlashInfer、FP8 KV 和容量。

### 失败分支

1. 退出码 4/超时：查看两个 smoke server log 和 Docker 状态，执行 diagnose，然后重跑同一 smoke；不要进入 probe。
2. OOM：确认没有其他 GPU 进程；不要降低固定 16G KV 预算来伪造主实验。
3. backend/dtype 日志缺失：当作硬失败；不要只相信 CLI 参数。
4. 两次结果不一致：MVP 的随机 token scale 不稳定。执行以下精确命令收集 dataset-calibrated KV-only scales 扩展所需输入，然后停止本 run：

```bash
python3 -m autokv diagnose --project-root "$AUTOKV_ROOT" --profile quick --json
python3 -m autokv status --project-root "$AUTOKV_ROOT" --profile quick --json
```

数据集标定扩展不属于这个小项目的已实现主路径；不要把 LLM Compressor 生成的新 checkpoint 塞进当前 run ID，也不要同时量化权重。正式面试应如实说“随机 token scale 的确定性 gate 未通过，因此没有进行事后挑选”。官方扩展入口是 [LLM Compressor KV-cache quantization examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache)。

---

## 阶段 10：在 tmux 中启动 quick，可分步复核

### 命令

启动一个继承当前 `AUTOKV_ROOT`、`PYTHONPATH` 和可选 `HF_TOKEN` 的 tmux：

```bash
tmux new-session -s autokv-quick
```

进入 tmux 后执行：

```bash
cd "$AUTOKV_ROOT"
set -o pipefail
python3 -m autokv run \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --port 8000 \
  --json 2>&1 | tee operator-logs/quick-run.log
printf 'quick_run_exit=%s\n' "${PIPESTATUS[0]}" | tee operator-logs/quick-run.exit
```

按 `Ctrl-b`，松开，再按 `d`，从 tmux detach。不要按 `Ctrl-c`，除非确实要中断当前配置。

如果希望面试演示每一阶段而不是一键运行，可用下面的等价分步命令；完成状态会复用，不能同时与 `run` 并发执行：

```bash
python3 -m autokv probe --project-root "$AUTOKV_ROOT" --profile quick --port 8000 --json
python3 -m autokv select --project-root "$AUTOKV_ROOT" --profile quick --json
python3 -m autokv evaluate --project-root "$AUTOKV_ROOT" --profile quick --port 8000 --json
python3 -m autokv benchmark --project-root "$AUTOKV_ROOT" --profile quick --port 8000 --json
python3 -m autokv report --project-root "$AUTOKV_ROOT" --profile quick --json
```

### 预期时间

quick 总 GPU 时间约 5–12 小时。首次 kernel compilation、磁盘速度和网络会影响时间；不要把估计当超时阈值。

### 成功判据

tmux session 存在；最终 `quick_run_exit=0`；`run` JSON 的 `complete=true`、`completed_steps` 含全部 9 阶段并给出 report/csv/svg 路径。

### 失败分支

1. `tmux: command not found`：让管理员安装 tmux，或使用组织认可的持久会话工具；不要在普通 SSH 前台冒险跑数小时任务。
2. 8000 端口被占：用 `ss -ltnp | grep ':8000'` 查明所有者。若是外部服务，整次实验统一改为 `--port 18000`；不要结束未知进程。
3. 某配置失败：`run` 保存已完成配置。不要删 `runs/`；进入阶段 11 诊断，然后以同 profile、root、port 重跑。
4. 不要开第二个 quick 进程；单卡并发运行会破坏显存和性能公平性。

---

## 阶段 11：查看状态、SSH 重连、读日志与恢复

### 命令

列出/连接 tmux：

```bash
tmux list-sessions
tmux attach-session -t autokv-quick
```

不进入 tmux也可查看只读状态：

```bash
python3 -m autokv status \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json | python3 -m json.tool
tail -n 100 operator-logs/quick-run.log
docker ps -a --filter label=io.autokv.project=autokv-skip
nvidia-smi
```

找到当前 run ID 与日志：

```bash
RUN_ID="$(python3 -m autokv status --project-root "$AUTOKV_ROOT" --profile quick --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"] or "")')"
printf 'RUN_ID=%s\n' "$RUN_ID"
find "runs/$RUN_ID" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail -n 80
grep -RHiE 'error|traceback|oom|out of memory|unsupported|failed' "runs/$RUN_ID" | tail -n 100 || true
```

生成诊断包：

```bash
python3 -m autokv diagnose \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json | tee operator-logs/diagnose-quick.json.stdout
```

中断或 SSH 重连后的恢复命令与原命令完全相同：

```bash
python3 -m autokv run --project-root "$AUTOKV_ROOT" --profile quick --port 8000 --json
```

### 预期时间

status 数秒；诊断包通常 1 分钟内；恢复只重做未完成配置。

### 成功判据

status 的已完成步骤逐步变为 true；不存在两个带项目 label 的运行中 server；诊断 tar.gz 位于 `runs/_diagnostics/` 且不含 `.cache`、模型文件或 token。

### 失败分支

1. SSH 断开：重新登录，重设阶段 1 的变量和阶段 4 的可选 token，再 attach tmux；不需要重跑已完成配置。
2. 主机重启导致容器停止：先确认无残留 AutoKV 容器，再执行同一 `run`。状态 SHA/行数不完整的精确配置会续跑。
3. 有外部容器：控制器会检查 label，拒绝停止非本项目容器。不要手工使用模糊容器名批量停止。
4. 单个 JSONL 被手工编辑后 hash 不符：保留证据并生成 diagnose；不要伪造 state hash。

---

## 阶段 12：生成报告、逐项验收并导出面试材料

即使 `run` 已生成报告，也再次调用 `report` 从已验证原始产物重建，证明统计与采集解耦。

### 命令

```bash
python3 -m autokv report \
  --project-root "$AUTOKV_ROOT" \
  --profile quick \
  --json | tee operator-logs/report-quick.json.stdout
python3 -m autokv status --project-root "$AUTOKV_ROOT" --profile quick --json | python3 -m json.tool
RUN_ID="$(python3 -m autokv status --project-root "$AUTOKV_ROOT" --profile quick --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"
REPORT_DIR="runs/$RUN_ID/report"
sed -n '1,260p' "$REPORT_DIR/REPORT.zh-CN.md"
column -s, -t < "$REPORT_DIR/summary.csv" | sed -n '1,40p'
test -s "$REPORT_DIR/capacity.svg"
sha256sum \
  runs/_environment/doctor.json \
  runs/_environment/lock.json \
  "runs/$RUN_ID/selection.json" \
  "$REPORT_DIR/REPORT.zh-CN.md" \
  "$REPORT_DIR/summary.csv" \
  "$REPORT_DIR/capacity.svg"
```

导出一个不含模型 cache 的面试包：

```bash
mkdir -p exports
tar -czf "exports/autokv-skip-$RUN_ID-interview.tar.gz" \
  README.md RUNBOOK.zh-CN.md configs/quick.json \
  runs/_environment/doctor.json runs/_environment/lock.json \
  "runs/$RUN_ID/selection.json" \
  "runs/$RUN_ID/probe/index.json" \
  "runs/$RUN_ID/quality/index.json" \
  "runs/$RUN_ID/perf/index.json" \
  "$REPORT_DIR"
sha256sum "exports/autokv-skip-$RUN_ID-interview.tar.gz"
tar -tzf "exports/autokv-skip-$RUN_ID-interview.tar.gz" | sed -n '1,120p'
```

### 预期时间

1–5 分钟。

### 成功判据

按以下清单逐项确认：

- 报告显示驱动 `535.230.02`、镜像 `@sha256:`、模型 40 位 revision、FlashInfer、16G；
- selection 恰好 4 个 Auto 层；5 个 Random-4 唯一，且没有重复 Auto/First/Last/Inverted；
- BF16、FP8、Auto-4 使用同一 dataset hash、镜像 digest、模型 revision；
- FP8 实测 KV capacity / BF16 是否达到 1.85×；Auto-4 是否达到 1.65×；
- 只有 `Q_bf16-Q_fp8>0.01` 才显示 gap recovery；
- Auto-4 是否高于 Random-4 中位数和 Inverted-4；
- 若未通过，报告明确写“未通过”或“未观察到足够缺口”，没有把噪声称为提升；
- 性能结论区分 TTFT、TPOT/ITL、吞吐与容量，不声称 A6000 有原生 FP8 Tensor Core；
- 导出包列表不含 `.cache`、safetensors、token 或模型权重。

### 失败分支

1. 报告命令退出 5：quality/perf 尚未完成，回阶段 11 恢复同一 run。
2. capacity 与理论偏差超过 10%：保留 server log，解释 page/block 对齐与 runtime reserve；不要改预算重跑出“好看”数值。
3. Auto-4 不胜随机：保留负结果，面试按 interview guide 的负结果版本讲；可选择 full 检查 coarse-to-fine 漏层假设。
4. CSV/SVG 缺失或为空：报告不完整，不导出；重新运行 report 并检查错误。

---

## 阶段 13：仅在条件满足时运行 full profile

full 扫描全部 32 个单层配置，用来检验 quick 的两级搜索是否漏掉分散敏感层。它不是为了在 quick 负结果后无限调参。

### 进入条件

同时满足以下条件才运行：quick 全部产物完整；服务器可连续占用 12–24 小时；磁盘仍有至少 50 GiB；你需要在面试前验证搜索近似，或 quick 的层信号有解释价值。若只是想完成一个小项目，quick 已足够。

### 命令

```bash
python3 -m autokv make-data --project-root "$AUTOKV_ROOT" --profile full --json
python3 -m autokv dry-run --project-root "$AUTOKV_ROOT" --profile full --json
python3 -m autokv smoke --project-root "$AUTOKV_ROOT" --profile full --port 8000 --json
tmux new-session -s autokv-full
```

进入 full tmux 后：

```bash
cd "$AUTOKV_ROOT"
set -o pipefail
python3 -m autokv run \
  --project-root "$AUTOKV_ROOT" \
  --profile full \
  --port 8000 \
  --json 2>&1 | tee operator-logs/full-run.log
printf 'full_run_exit=%s\n' "${PIPESTATUS[0]}" | tee operator-logs/full-run.exit
```

状态与报告：

```bash
python3 -m autokv status --project-root "$AUTOKV_ROOT" --profile full --json | python3 -m json.tool
python3 -m autokv report --project-root "$AUTOKV_ROOT" --profile full --json
```

### 预期时间

12–24 小时；full dry-run 应显示 34 个核心配置、每配置 18 个 probe 样本、45 个 final 样本和 18 个 benchmark 场景。

### 成功判据

full 有独立 dataset hash 和 run ID，但必须复用同一个已锁镜像 digest 与模型 revision；报告明确 selection scope 是 all-32-layers。比较 quick/full 时只比较选择稳定性与各自完整实验，不把两个 profile 的不同样本直接拼成一个置信区间。

### 失败分支

1. full 中断：使用同一 full `run` 恢复；不要删 quick。
2. full 选层与 quick 不同：这是 coarse-to-fine 漏层证据，报告为近似误差，不用事后重定义 quick。
3. full 没有带来更好 Auto-4：保留结果，说明层交互或代理不稳定；停止追加搜索预算。

---

## 故障时绝对不要做的事

- 不安装、升级、降级、卸载任何 NVIDIA GPU driver 包；
- 不运行 CUDA `.run` installer，不重建 DKMS，不修改 Secure Boot；
- 不把宿主 CUDA 目录挂进容器；
- 不使用高权限容器绕过 device/runtime gate；
- 不执行宽泛 Docker 清理，不批量删除 cache；
- 不编辑 `runs/*/*.state.json`、结果 JSONL 或 hash 来跳过失败；
- 不把主镜像失败误判成“版本越旧越兼容”；只有程序分类允许时才回退；
- 不并行运行两个配置，不在不同镜像版本间拼性能表。

## 权威兼容依据

- [vLLM：旧 NVIDIA driver 的官方 Docker CUDA compatibility 模式](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM：CUDA/PTX 兼容问题排查](https://docs.vllm.ai/en/stable/usage/troubleshooting/)
- [NVIDIA：CUDA forward compatibility 支持矩阵](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)
- [NVIDIA Container Toolkit 安装与 Docker 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Engine 官方安装入口](https://docs.docker.com/engine/install/)

本手册把兼容性当作实测 gate，而不是仅凭版本表推断。只要阶段 5 或阶段 9 不通过，就没有正式 AutoKV-Skip 结果。
