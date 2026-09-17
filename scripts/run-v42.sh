#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AUTOKV_VLLM_BIN:?请设置既有 vLLM 可执行文件路径}"
: "${AUTOKV_MODEL_PATH:?请设置服务器本地模型目录}"
: "${AUTOKV_FP4_SITE:?请指向 v4.1 已运行成功的隔离 site 目录}"
export PYTHONPATH="$PWD:$AUTOKV_FP4_SITE${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_USE_TRTLLM_ATTENTION=0
export FLASHINFER_CUDA_ARCH_LIST=8.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "$(dirname "$AUTOKV_VLLM_BIN")/python" -m autokv v4-run --project-root "$PWD" --config configs/v4.2/quality.json "$@"
