# v4.1 离线运行手册

本版使用 A6000 的 FP4 KV 软件解码路径，需要一次性准备新版 FlashInfer 和隔离 vLLM 副本。代码与 CPU 测试不能代替该服务器的 GPU 验证。不要直接在正在跑 v4.0 的目录或 Python 安装中覆盖依赖。

服务器有 Git，但不能访问外网或推送；Linux 登录设备能联网但不能推送。依赖、代码从 Linux 传入，结果从服务器传回 Linux，再用 GitHub 网页提交。启动实验不要求 Git 提交、推送、doctor、smoke 或人工核对哈希。

## 1. 联网 Linux：准备代码和 FlashInfer

v4.1 发布到 GitHub 后，可从网页下载该分支 ZIP；发布前使用本次工作区导出的源码。将其解压到新的 `autokv-skip-v4.1` 目录。以下编包命令使用 Linux x86_64 的 Python 3.12，与服务器一致；无需登录 GitHub 或推送。

```bash
export AUTOKV_SSH='用户名@服务器地址'
export AUTOKV_TRANSFER="$(mktemp -d "$HOME/autokv-v41-transfer.XXXXXX")"
mkdir -p "$AUTOKV_TRANSFER/wheels"
git clone https://github.com/flashinfer-ai/flashinfer.git "$AUTOKV_TRANSFER/flashinfer"
cd "$AUTOKV_TRANSFER/flashinfer"
git checkout a4c17d77a15fbff1ef7b34a4ad2b5faaa5bdd6dc
git submodule update --init --recursive
BUILD_NVEP=0 python3.12 -m pip wheel --no-deps . -w "$AUTOKV_TRANSFER/wheels"
```

不要下载旧版 `0.6.16.post3` 作为替代：它不是本版选定的 FA2 软件解码实现。源码 wheel 包含 JIT 所需的 include/csrc 和子模块头文件；CUDA 内核首次在服务器用既有 CUDA 12.9 编译。

将 Python 依赖一并放入离线包，保留既有 torch 和 CUDA 主版本。下面准备 FlashInfer 的直接依赖；目标环境沿用已经能运行 v4.0 的 torch/vLLM 及其依赖，不是从空环境安装。

```bash
python3.12 - <<'PY'
from pathlib import Path
lines = Path('requirements.txt').read_text().splitlines()
lines = [s for s in lines if s.strip() and not s.startswith('#') and s.strip() != 'torch']
lines = ['cuda-python>=12.0,<13' if s.startswith('cuda-python') else s for s in lines]
Path('runtime-without-torch.txt').write_text('\n'.join(lines)+'\n')
PY
python3.12 -m pip download --no-deps --only-binary=:all: --pre \
  -r runtime-without-torch.txt -d "$AUTOKV_TRANSFER/wheels"
cp runtime-without-torch.txt "$AUTOKV_TRANSFER/"
cd "$AUTOKV_TRANSFER"
tar -czf fp4-runtime.tar.gz wheels runtime-without-torch.txt
scp fp4-runtime.tar.gz "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/"
```

这一步不升级服务器驱动、不替换 torch。若新增 Python 包需要旧环境没有的传递依赖，应在联网设备下载对应 Linux/Python 3.12 wheel 后一并传入；不要让服务器尝试在线安装。新 FlashInfer 与现有 torch/CUDA 的完整组合尚未在本项目目标机验证，编译或导入失败属于后端适配诊断，不能记成模型答错。

将项目源码目录整体复制或打包传到服务器新目录；同时复用原来的模型和离线来源，不需要重新下载模型或 SQuAD/Hotpot 数据。

## 2. 离线服务器：安装独立副本

以下在新项目目录执行。`AUTOKV_VLLM_BIN` 指向原来可运行的 vLLM，模型路径填写原路径。默认路径示例需要按实际目录调整。

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip-v4.1
export AUTOKV_VLLM_BIN=/mnt_d/huangxiaoyuan/autokv-skip/.venv-vllm-pgcg/bin/vllm
export AUTOKV_MODEL_PATH='/已有模型目录/Mistral-7B-Instruct-v0.3'
export AUTOKV_PYTHON="$(dirname "$AUTOKV_VLLM_BIN")/python"
mkdir -p .runtime/v41/offline
tar -xzf /mnt_d/huangxiaoyuan/fp4-runtime.tar.gz -C .runtime/v41/offline
"$AUTOKV_PYTHON" -m pip install --no-index --no-deps \
  --find-links .runtime/v41/offline/wheels --target .runtime/v41/site \
  -r .runtime/v41/offline/runtime-without-torch.txt
"$AUTOKV_PYTHON" -m pip install --no-index --no-deps \
  --find-links .runtime/v41/offline/wheels --target .runtime/v41/site flashinfer-python
"$AUTOKV_PYTHON" scripts/prepare-v41-runtime.py
```

安装脚本复制原 vLLM 包，再在副本里修改 FlashInfer 接口；原环境不改动。生成的差异文件为 `.runtime/v41/site/vllm-fp4.patch`。若原 vLLM 与 `c18d29d36` 接口不同，脚本会说明匹配失败的位置，不会猜测着覆盖源码。重复安装请使用新的副本目录，并通过 `AUTOKV_FP4_SITE` 指向它。

如果 v4.0 使用了自定义 CUDA/动态库路径，将旧项目 `runs/_environment/lock.json` 复制到新项目同一路径；`AUTOKV_MODEL_PATH` 和 `AUTOKV_VLLM_BIN` 会覆盖其中对应位置。FP4 的 JIT 缓存会自动放到新目录，不复用旧版编译结果。

## 3. 可选：一次性 GPU 数值诊断

新增 FP4 后端首次部署时，可以先比较真实分页 attention 与参考结果。它是针对新存储格式的数值诊断，不是每次实验都必须重复的门禁；主运行不会自动插入这一步。

```bash
export AUTOKV_FP4_SITE="$PWD/.runtime/v41/site"
export PYTHONPATH="$AUTOKV_FP4_SITE:$PWD"
export VLLM_KV_CACHE_LAYOUT=HND
export FLASHINFER_CUDA_ARCH_LIST=8.6
export FLASHINFER_CUBIN_DIR="$PWD/.runtime/v41/cubins"
export FLASHINFER_WORKSPACE_BASE="$PWD/.runtime/v41/flashinfer-cache"
mkdir -p runs
"$AUTOKV_PYTHON" -m scripts.check_v41_fp4 2>&1 | tee runs/v41-fp4-numerics.log
```

通过会保存 `runs/v41-fp4-numerics.json`；失败保留日志。这只验证 FP4 写入、非 1 scale、prefill/decode 的数值，不代表 Mistral 质量和混合层容量已通过。

## 4. 正式运行

配置位于 `configs/v4.1/quality.json`。默认 BF16 ≥ 0.80、BF16−FP4 严格 > 0.10；要使用 0.20，在新运行前修改 `construction.min_fp4_gap`。最终质量容差没有改变。输入种子和独立测试不能依据本轮测试表现调整。

先结束或暂停占用同一 GPU 的旧实验，再启动本轮，避免显存和性能互相影响。首次 JIT 编译可能需要几分钟。

```bash
export AUTOKV_DATA_SOURCE='/已有v3或v4项目/data/v3.0/source'
mkdir -p runs
set -o pipefail
bash scripts/run-v41.sh --source-dir "$AUTOKV_DATA_SOURCE" --json \
  2>&1 | tee runs/v41-cli.log
printf '%s\n' "${PIPESTATUS[0]}" > runs/v41-cli.exitcode
```

只做构造可加 `--construction-only`。续跑使用日志中的 `v41-...` ID；省略 `--construction-only` 将继续选层和独立测试：

```bash
bash scripts/run-v41.sh --run-id v41-实际运行ID --json \
  2>&1 | tee -a runs/v41-cli.log
```

默认先跑两端的 discovery 和 confirmation。确认成功后冻结一个规则，生成新实验集/测试集，接续束搜索。构造失败完整结束时是负结果；运行错误是未完成。P0 在正式实验集已达标时仍允许早停。

检查 `runs/v41-*/completed-manifest.json`、`construction/result.json` 和 `report/`。报告文件名沿用 schema 4 的 `CONSTRUCTION-v4.zh-CN.md`、`QUALITY-v4.zh-CN.md`，正文标明 v4.1 和 FP4。真实容量以服务日志与结果记录为准，不用 3.56 倍理论值替代。

## 5. 回传和 GitHub 网页提交

成功、失败、中断都可以导出，导出不要求测试通过。服务器执行：

```bash
"$AUTOKV_PYTHON" -m scripts.export_v41_results --run-id v41-实际运行ID
```

若服务在创建运行 ID 前就失败，省略 `--run-id`，导出已保存的 `runs/v41-cli.log`。程序生成 `results/v41-时间/`，大归档自动按 20 MiB 分片。

在 Linux 登录设备执行，将刚才输出的实际目录名代入：

```bash
scp -r "$AUTOKV_SSH:/mnt_d/huangxiaoyuan/autokv-skip-v4.1/results/v41-实际时间" ./
```

在 Linux 浏览器打开仓库，选择 **v4.1** 分支，使用 `Add file → Upload files`，上传该结果目录中的 README、报告和全部归档分片，填写说明并手动提交。不在服务器或 Linux 设备执行 `git push`。
