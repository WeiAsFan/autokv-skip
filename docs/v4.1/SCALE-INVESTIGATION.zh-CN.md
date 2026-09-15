# 自动 scale 导致乱码的调查

**结论：发现了一个有直接代码依据、足以破坏 attention 的缺陷，但无法确定它就是此前乱码的历史根因。** 当时启用自动 scale 的失败日志、逐层 scale 和首个异常输出未保留；不能据现有材料排除其他问题。

## 已确认的代码事实

既有运行记录使用 Mistral 7B、A6000 SM86、vLLM `0.1.dev19475+gc18d29d36`、FlashInfer `0.6.16.post3`。归档的成功 FP8 日志显示 prefill 和 decode 的 Q 均为 BF16；这些是固定 scale=1 的成功记录，不是失败复现。

[vLLM attention](https://github.com/vllm-project/vllm/blob/c18d29d36/vllm/model_executor/layers/attention/attention.py) 的自动估计同时计算 Q/K/V scale：`q_scale=max(abs(Q))/q_range` 等；随后关闭该层的计算标志，只计算一次。它并不持续重新估计每个 token 的 scale。

[vLLM FlashInfer 后端](https://github.com/vllm-project/vllm/blob/c18d29d36/vllm/v1/attention/backends/flashinfer.py) 在 SM86 选择 BF16 Q，但原来的 paged prefill/decode 调用仍无条件传入 `layer._q_scale_float`。[FlashInfer v0.6.16 的 paged prefill](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16/flashinfer/prefill.py) 会将该值乘入 `sm_scale`，而不根据 Q dtype 自动取消。这里核对的是 v0.6.16 标签源码，不是当年服务器 post3 安装目录的逐字节副本。

若 K 量化为 `Kq≈K/sk`，BF16 Q 未除以 `sq`，正确分数为：

```text
Q × Kqᵀ × sk / sqrt(d)
```

错误地再乘 `sq` 会改变 softmax 温度。例如两个 logit 为 `[4,0]` 时，首项概率约 0.982；若额外乘 0.01，就变成约 0.510。跨层注意力可能因此接近均匀，模型输出可严重失真。将 scale 设为 1 恰好隐藏这种错误，但这个现象本身不能唯一定位根因。

v4.1 的隔离后端已按实际 Q dtype 传值：BF16 Q 使用 `q_scale=1`；真实 FP8 Q 才使用反量化 scale。K/V 的 scale 仍正常传递，不能为修 Q 而取消 K/V 反量化。

## 仍未确定的部分

未得到实际失败时的 `q_scale/k_scale/v_scale`，不能证明当时 `q_scale` 确实异常，也不能证明其服务代码与上述路径完全一致。一次性估计的输入是否代表真实数据、是否存在零/非有限 scale、K/V 是否饱和、具体内核是否正确使用非 1 scale，都未由历史证据排除。

因此，“自动 scale 本身不适合 Mistral”或“自动校准必然乱码”都不是现有证据支持的结论。准确表述是：存在 BF16 Q 被错误再次缩放的代码风险；修复针对这一风险，但历史根因仍未闭环。

## 如需复现旧问题

使用同一短提示、同一 seed、同一原环境依次比较：固定 scale=1；旧后端自动 scale；仅修复 BF16 Q scale 后的自动 scale。保存服务完整日志、每层 Q/K/V scale、Q 实际 dtype、首批输出和是否发生截断。若第三种恢复正常且 K/V scale 一致，才形成较强因果证据；否则继续检查 K/V 写入和读取。

该诊断不是 v4.1 正式实验的前置程序。FP4 使用随每组 KV 保存的局部 scale，本版不重新启用首次全局自动 scale。
