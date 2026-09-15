"""为已部署的 c18d29d36 vLLM 制作隔离副本，不修改 v4.0 环境。"""
from __future__ import annotations

import ast
import difflib
import importlib.util
import shutil
from pathlib import Path

FLASHINFER_COMMIT = "a4c17d77a15fbff1ef7b34a4ad2b5faaa5bdd6dc"
BACKEND = Path("v1/attention/backends/flashinfer.py")


def patch_backend(source):
    """按接口片段匹配，拒绝把补丁写到接口不同的版本；不要求人工核对哈希。"""
    def change(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError("vLLM FlashInfer 接口与 c18d29d36 不同，未写入补丁："+old[:100])
        source = source.replace(old, new)

    change('FP4_DTYPE = torch.uint8', '''FP4_DTYPE = torch.uint8


def _autokv_sm86():
    return current_platform.is_device_capability(86)
''')
    change('if kv_cache_dtype == "nvfp4":\n            return (',
           'if kv_cache_dtype == "nvfp4":\n            if _autokv_sm86():\n                return True\n            return (')
    change('''                if (
                    force_use_trtllm_attention() is False''', '''                if not _autokv_sm86() and (
                    force_use_trtllm_attention() is False''')
    change('self.kv_cache_dtype = self.cache_dtype',
           'self.kv_cache_dtype = torch.uint8 if _autokv_sm86() else self.cache_dtype')
    change('''        if cache_dtype == "nvfp4":
            return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")''',
           '''        if cache_dtype == "nvfp4":
            if _autokv_sm86():
                return self.model_config.dtype
            return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")''')
    change('backend = "trtllm-gen" if self.is_kvcache_nvfp4 else "auto"',
           'backend = ("fa2" if _autokv_sm86() else "trtllm-gen") if self.is_kvcache_nvfp4 else "auto"', 2)
    change('FP8_DTYPE if self.is_kvcache_nvfp4 else self.model_config.dtype',
           'FP8_DTYPE if self.is_kvcache_nvfp4 and not _autokv_sm86() else self.model_config.dtype', 2)
    change('self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE',
           'self.is_kvcache_nvfp4 and not _autokv_sm86() and output.dtype != FP8_DTYPE', 4)
    change('if self.is_kvcache_nvfp4 and vllm_config is not None:',
           'if self.is_kvcache_nvfp4 and not _autokv_sm86() and vllm_config is not None:')
    # BF16 Q 未量化，不能再乘 FP8 q_scale；分别按实际 prefill/decode dtype 判断。
    change('''                        prefill_query,
                        kv_cache_for_fi,
                        q_scale=layer._q_scale_float,''', '''                        prefill_query,
                        kv_cache_for_fi,
                        q_scale=layer._q_scale_float if prefill_query.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 1.0,''')
    change('''                        decode_query,
                        kv_cache_for_fi,
                        q_scale=layer._q_scale_float,''', '''                        decode_query,
                        kv_cache_for_fi,
                        q_scale=layer._q_scale_float if decode_query.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 1.0,''', 2)
    change('''            if self.is_kvcache_nvfp4:
                # (B, 2*H, N, full_dim)''', '''            if self.is_kvcache_nvfp4 and _autokv_sm86():
                from autokv.fp4_cache import write_cache
                write_cache(key, value, kv_cache, slot_mapping,
                            self.num_kv_heads, self.head_size,
                            layer._k_scale_float, layer._v_scale_float)
                return
            if self.is_kvcache_nvfp4:
                # (B, 2*H, N, full_dim)''')
    ast.parse(source)
    return source


def prepare_overlay(destination, base_package=None):
    destination = Path(destination).resolve()
    if base_package is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None:
            raise ValueError("请使用既有 vLLM 环境的 Python 执行安装")
        base_package = Path(spec.origin).parent
    source = Path(base_package).resolve()
    target = destination / "vllm"
    if target.exists() or source == target or source in target.parents:
        raise ValueError("目标需为独立的新 vLLM 副本；请指定新的输出目录")
    before = (source / BACKEND).read_text(encoding="utf-8")
    after = patch_backend(before)
    fi_env = destination / "flashinfer/jit/env.py"
    if not fi_env.exists():
        raise ValueError("请先将新版 FlashInfer wheel 离线安装到同一 --output 目录")
    fi_before = fi_env.read_text(encoding="utf-8")
    marker = 'def _get_aot_dir():\n'
    if fi_before.count(marker) != 1:
        raise ValueError("FlashInfer 的 JIT 缓存接口与指定版本不同")
    # 独立运行只用新源码 JIT，避免加载旧环境的 flashinfer-jit-cache 二进制。
    fi_after = fi_before.replace(marker, marker+'    if os.getenv("AUTOKV_FP4_SITE"):\n        return _package_root / "data" / "aot"\n')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / BACKEND).write_text(after, encoding="utf-8")
    fi_env.write_text(fi_after, encoding="utf-8")
    (destination / "vllm-fp4.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=str(BACKEND), tofile=str(BACKEND))), encoding="utf-8")
    return target
