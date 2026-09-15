"""A6000 的 E2M1 KV 写入：持久缓存只有打包数据与每 16 元素 scale。

只对本次新增 token 做量化；attention 由 FlashInfer FA2 在片上展开历史 KV。
使用普通 PyTorch 运算，避免依赖 SM100 的 FP4 转换指令。首版使用 eager。
"""
from __future__ import annotations


def quantize(values, global_scale=1.0):
    import torch

    if values.shape[-1] % 16:
        raise ValueError("FP4 分组要求末维为 16 的倍数")
    if not global_scale > 0:
        raise ValueError("全局 scale 必须为正数")
    shape = values.shape
    x = values.float().reshape(*shape[:-1], shape[-1] // 16, 16) / global_scale
    # E4M3 最小正数 2^-9；先舍入 scale，再用实际存储值量化。
    scale = (x.abs().amax(-1) / 6).clamp(2**-9, 448).to(torch.float8_e4m3fn)
    normalized = x / scale.float().unsqueeze(-1)
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude = normalized.abs().contiguous()
    index = torch.bucketize(magnitude, boundaries)
    # E2M1 的 halfway ties 取编码末位为偶数者。
    tie_up = (index % 2 == 1) & (magnitude == boundaries[index.clamp(max=6)])
    nibble = (index + tie_up + 8 * (normalized < 0)).to(torch.uint8).reshape(shape)
    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    return packed, scale


def dequantize(packed, scales, global_scale=1.0):
    """数值验证用参考解码；运行时 attention 不调用此函数。"""
    import torch

    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    nibble = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = table[nibble & 7] * (1 - 2 * (nibble >> 3))
    return values * scales.float().repeat_interleave(16, dim=-1) * global_scale


def cache_views(cache, num_heads, head_dim):
    """与 vLLM NVFP4 [K 数据|K scales|V 数据|V scales] 每页布局一致。"""
    import torch

    pages, heads2, page_size, full_dim = cache.shape
    if cache.dtype != torch.uint8 or heads2 != 2*num_heads or full_dim != head_dim//2+head_dim//16:
        raise ValueError("NVFP4 缓存的形状或类型不匹配")
    if cache.stride(3) != 1 or cache.stride(2) != full_dim or cache.stride(1) != page_size*full_dim:
        raise ValueError("v4.1 写入只支持 HND 缓存布局")
    sides = []
    for side in cache.split(num_heads, dim=1):
        offset = side.storage_offset()
        data_dim, sf_dim = head_dim//2, head_dim//16
        data = side.as_strided((pages, num_heads, page_size, data_dim),
                              (cache.stride(0), page_size*data_dim, data_dim, 1), offset)
        scales = side.as_strided((pages, num_heads, page_size, sf_dim),
                                (cache.stride(0), page_size*sf_dim, sf_dim, 1),
                                offset+num_heads*page_size*data_dim).view(torch.float8_e4m3fn)
        sides.append((data, scales))
    return sides


def write_cache(key, value, cache, slots, num_heads, head_dim, k_scale=1.0, v_scale=1.0):
    import torch

    views = cache_views(cache, num_heads, head_dim)
    slots = slots.flatten().long()
    valid = slots >= 0
    locations = slots[valid]
    pages, offsets = locations // cache.shape[2], locations % cache.shape[2]
    heads = torch.arange(num_heads, device=cache.device)[None, :]
    for tensor, (data, scales), scale in zip((key, value), views, (k_scale, v_scale)):
        current = tensor.reshape(-1, num_heads, head_dim)[:slots.numel()][valid]
        packed, sf = quantize(current, scale)
        data[pages[:, None], heads, offsets[:, None]] = packed
        # 以字节写入，避免旧版 PyTorch 不支持 Float8 的 index_put。
        scales.view(torch.uint8)[pages[:, None], heads, offsets[:, None]] = sf.view(torch.uint8)
