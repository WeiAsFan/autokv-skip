"""离线服务器：复制现有 vLLM 并接入 A6000 FP4，不修改原安装。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autokv.fp4_runtime import prepare_overlay

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=".runtime/v41/site")
parser.add_argument("--base-package", help="可选：既有 vLLM 包目录")
args = parser.parse_args()
print(prepare_overlay(args.output, args.base_package))
