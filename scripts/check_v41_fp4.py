"""可选 GPU 数值验证：实际 FP4 分页写入与 FA2 prefill/decode 对齐参考 attention。"""
import json
from pathlib import Path

import torch
import flashinfer
from autokv.fp4_cache import cache_views, dequantize, write_cache


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("该验证需要实验服务器上的 GPU")
    torch.manual_seed(42)
    device = "cuda"
    heads, qo_heads, dim, page_size, length = 8, 32, 128, 16, 259
    cache = torch.zeros((17, 2*heads, page_size, 72), dtype=torch.uint8, device=device)
    k, v = [torch.randn(length, heads, dim, dtype=torch.bfloat16, device=device)*s for s in (1.3, .7)]
    slots = torch.arange(length, device=device)
    write_cache(k, v, cache, slots, heads, dim, .5, 2.)
    (kd, ks), (vd, vs) = cache_views(cache, heads, dim)
    ref_k = dequantize(kd, ks, .5).permute(0, 2, 1, 3).reshape(-1, heads, dim)[:length].repeat_interleave(4, 1)
    ref_v = dequantize(vd, vs, 2.).permute(0, 2, 1, 3).reshape(-1, heads, dim)[:length].repeat_interleave(4, 1)
    indptr = torch.tensor([0, 17], dtype=torch.int32, device=device)
    indices = torch.arange(17, dtype=torch.int32, device=device)
    last = torch.tensor([3], dtype=torch.int32, device=device)
    checks = []
    for qo_len in (1, 17, 65):
        q = torch.randn(qo_len, qo_heads, dim, dtype=torch.bfloat16, device=device)
        workspace = torch.empty(128*1024*1024, dtype=torch.uint8, device=device)
        common = dict(num_qo_heads=qo_heads, num_kv_heads=heads, page_size=page_size,
                      q_data_type=torch.bfloat16, kv_data_type=torch.uint8, o_data_type=torch.bfloat16)
        if qo_len == 1:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND", use_tensor_cores=True, backend="fa2")
            wrapper.plan(indptr, indices, last, head_dim=dim, **common)
        else:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "HND", backend="fa2")
            wrapper.plan(torch.tensor([0, qo_len], dtype=torch.int32, device=device), indptr, indices, last,
                         head_dim_qk=dim, causal=True, **common)
        output = wrapper.run(q, (kd, vd), kv_cache_sf=(ks, vs), q_scale=1., k_scale=.5, v_scale=2.)
        logits = torch.einsum("qhd,khd->hqk", q.float(), ref_k)/(dim**.5)
        mask = torch.arange(length, device=device)[None, :] > (length-qo_len+torch.arange(qo_len, device=device)[:, None])
        logits.masked_fill_(mask[None], float("-inf"))
        reference = torch.einsum("hqk,khd->qhd", logits.softmax(-1), ref_v)
        error = float((output.float()-reference).abs().max())
        torch.testing.assert_close(output.float(), reference, atol=.02, rtol=.02)
        checks.append({"query_tokens": qo_len, "max_abs_error": error})
    result = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "flashinfer": flashinfer.__version__, "checks": checks,
              "note": "验证的是写入及 attention 数值；不代替模型实验或容量实测"}
    path = Path("runs/v41-fp4-numerics.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
