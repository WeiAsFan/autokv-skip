#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AUTOKV_VLLM_BIN:?请设置既有 vLLM 可执行文件路径}"
: "${AUTOKV_MODEL_PATH:?请设置服务器本地模型目录}"
export AUTOKV_FP4_SITE="${AUTOKV_FP4_SITE:-$PWD/.runtime/v41/site}"
if [[ ! -f "$AUTOKV_FP4_SITE/vllm-fp4.patch" || ! -d "$AUTOKV_FP4_SITE/flashinfer" ]]; then
    echo '请先按 docs/v4.1/RUNBOOK.zh-CN.md 安装隔离的 FP4 运行环境。' >&2
    exit 1
fi
export PYTHONPATH="$AUTOKV_FP4_SITE:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_USE_TRTLLM_ATTENTION=0
export FLASHINFER_CUDA_ARCH_LIST=8.6
export FLASHINFER_CUBIN_DIR="$PWD/.runtime/v41/cubins"
export FLASHINFER_WORKSPACE_BASE="$PWD/.runtime/v41/flashinfer-cache"
export TORCH_EXTENSIONS_DIR="$PWD/.runtime/v41/torch-extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "$(dirname "$AUTOKV_VLLM_BIN")/python" -m autokv v4-run --project-root "$PWD" --config configs/v4.1/quality.json "$@"
