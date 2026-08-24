"""KV-cache memory formulas used by the experiment and report."""


def kv_bytes_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    values = (num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError("KV dimensions and dtype bytes must be positive")
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def mixed_kv_bytes_per_token(
    num_layers: int,
    bf16_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    if num_layers <= 0 or not 0 <= bf16_layers <= num_layers:
        raise ValueError("bf16_layers must be between zero and num_layers")
    if num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError("KV dimensions must be positive")
    return 2 * num_kv_heads * head_dim * (num_layers + bf16_layers)


def ideal_capacity_gain(num_layers: int, bf16_layers: int) -> float:
    if num_layers <= 0 or not 0 <= bf16_layers <= num_layers:
        raise ValueError("bf16_layers must be between zero and num_layers")
    return (2 * num_layers) / (num_layers + bf16_layers)
