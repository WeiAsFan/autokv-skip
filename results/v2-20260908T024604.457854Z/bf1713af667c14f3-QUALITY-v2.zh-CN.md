# AutoKV-Skip v2.1 质量与容量报告

- 运行 ID：`bf1713af667c14f3`
- 状态：`passed`
- Calibration 候选：P1，BF16 层 [17]
- 最终策略：P1，BF16 层 [17]
- 主实验达标：是
- 留出集质量达标：是
- 容量验收：`passed`，目标至少 1.5×
- 实际源码、配置、数据和运行路径副本：`inputs/`
- KV 显存预算：16G；prefix caching 显式关闭；模型权重精度保持原样。

P1 在 calibration 与 held-out 满足预定质量约束。 实测 KV 容量达到目标。

`complete` 表示本次流程已给出结果，不等于主实验达标。

## Calibration 端点

| 策略 | k | Easy | Hard | Natural | S_v2 |
|---|---:|---:|---:|---:|---:|
| p32 | 32 | 1.0000 | 0.4769 | 0.4667 | 0.6478 |
| p0 | 0 | 0.6667 | 0.4583 | 0.4667 | 0.5306 |

| 策略 | 任务 | 绝对分数 |
|---|---|---:|
| p32 | aggregation_extraction | 0.3889 |
| p32 | hotpotqa_e | 0.6000 |
| p32 | multi_key_value | 0.9167 |
| p32 | niah | 1.0000 |
| p32 | qasper_e | 0.3333 |
| p32 | variable_tracking | 0.1250 |
| p0 | aggregation_extraction | 0.3333 |
| p0 | hotpotqa_e | 0.6000 |
| p0 | multi_key_value | 0.9167 |
| p0 | niah | 0.6667 |
| p0 | qasper_e | 0.3333 |
| p0 | variable_tracking | 0.1250 |

## Held-out

| 策略 | k | Easy | Hard | Natural | S_v2 |
|---|---:|---:|---:|---:|---:|
| p32 | 32 | 1.0000 | 0.5582 | 0.7584 | 0.7722 |
| p0 | 0 | 1.0000 | 0.5185 | 0.6919 | 0.7368 |
| layer-17-p1 | 1 | 1.0000 | 0.5926 | 0.7584 | 0.7837 |

| 策略 | 任务 | 绝对分数 |
|---|---|---:|
| p32 | aggregation_extraction | 0.5556 |
| p32 | hotpotqa_e | 0.6667 |
| p32 | multi_key_value | 0.9524 |
| p32 | niah | 1.0000 |
| p32 | qasper_e | 0.8501 |
| p32 | variable_tracking | 0.1667 |
| p0 | aggregation_extraction | 0.5556 |
| p0 | hotpotqa_e | 0.6667 |
| p0 | multi_key_value | 0.9167 |
| p0 | niah | 1.0000 |
| p0 | qasper_e | 0.7172 |
| p0 | variable_tracking | 0.0833 |
| layer-17-p1 | aggregation_extraction | 0.7778 |
| layer-17-p1 | hotpotqa_e | 0.6667 |
| layer-17-p1 | multi_key_value | 0.9167 |
| layer-17-p1 | niah | 1.0000 |
| layer-17-p1 | qasper_e | 0.8501 |
| layer-17-p1 | variable_tracking | 0.0833 |

## 配对差与不确定性

差值方向按名称中的左项减右项；区间用于解释不确定性，不参与运行门禁。

| 比较 | 类别 | 分差 | 95% CI |
|---|---|---:|---|
| calibration_P32-P0 | easy | +0.333333 | [+0.000000, +1.000000] |
| calibration_P32-P0 | hard | +0.018519 | [-0.027778, +0.069560] |
| calibration_P32-P0 | natural | +0.000000 | [+0.000000, +0.000000] |
| calibration_P32-P0 | global | +0.117284 | [-0.004630, +0.333333] |
| heldout_P32-P0 | easy | +0.000000 | [+0.000000, +0.000000] |
| heldout_P32-P0 | hard | +0.039683 | [+0.000000, +0.095238] |
| heldout_P32-P0 | natural | +0.066479 | [+0.000000, +0.199436] |
| heldout_P32-P0 | global | +0.035387 | [+0.000000, +0.083675] |
| heldout_P32-candidate | easy | +0.000000 | [+0.000000, +0.000000] |
| heldout_P32-candidate | hard | -0.034392 | [-0.210317, +0.091369] |
| heldout_P32-candidate | natural | +0.000000 | [+0.000000, +0.000000] |
| heldout_P32-candidate | global | -0.011464 | [-0.070106, +0.030456] |
| heldout_candidate-P0 | easy | +0.000000 | [+0.000000, +0.000000] |
| heldout_candidate-P0 | hard | +0.074074 | [+0.000000, +0.222222] |
| heldout_candidate-P0 | natural | +0.066479 | [+0.000000, +0.199436] |
| heldout_candidate-P0 | global | +0.046851 | [+0.000000, +0.118393] |

## 质量验收

全局下降 ≤ 0.01，Hard/Natural 各下降 ≤ 0.02；BF16 与候选基础题均须全过。

| 检查 | 通过 |
|---|---|
| reference_valid | 是 |
| global | 是 |
| hard | 是 |
| natural | 是 |
| easy | 是 |

## 已评估候选

只在 calibration 上选择；同预算按总分高、层号小排序。每个层集合在同一 split 只运行一次。

| 策略 | BF16 层 | k | S_v2 | 质量通过 |
|---|---|---:|---:|---|
| group-00-p4 | [0, 1, 2, 3] | 4 | 0.6185 | 否 |
| group-01-p4 | [4, 5, 6, 7] | 4 | 0.6673 | 否 |
| group-02-p4 | [8, 9, 10, 11] | 4 | 0.5074 | 否 |
| group-03-p4 | [12, 13, 14, 15] | 4 | 0.5812 | 否 |
| group-04-p4 | [16, 17, 18, 19] | 4 | 0.6386 | 否 |
| group-05-p4 | [20, 21, 22, 23] | 4 | 0.5259 | 否 |
| group-06-p4 | [24, 25, 26, 27] | 4 | 0.5136 | 否 |
| group-07-p4 | [28, 29, 30, 31] | 4 | 0.5074 | 否 |
| layer-04-p1 | [4] | 1 | 0.5686 | 否 |
| layer-05-p1 | [5] | 1 | 0.6231 | 否 |
| layer-06-p1 | [6] | 1 | 0.6216 | 否 |
| layer-07-p1 | [7] | 1 | 0.6139 | 否 |
| layer-16-p1 | [16] | 1 | 0.5136 | 否 |
| layer-17-p1 | [17] | 1 | 0.6478 | 是 |
| layer-18-p1 | [18] | 1 | 0.6386 | 否 |
| layer-19-p1 | [19] | 1 | 0.5429 | 否 |

## KV 容量

| 策略 | k | bytes/token | 理论倍率 | 实测 tokens | 实测倍率 |
|---|---:|---:|---:|---:|---:|
| p32 | 32 | 131072 | 1.0000× | 131072 | 1.0000× |
| p0 | 0 | 65536 | 2.0000× | 262144 | 2.0000× |
| layer-17-p1 | 1 | 67584 | 1.9394× | 254176 | 1.9392× |

## 结论范围

质量判定仅覆盖这份固定留出集与当前模型/运行时。选择的是已评估候选中的较小 BF16 预算，不保证全局最优。容量指同样 KV 显存预算下的 KV token 数，不等于模型上下文窗口或吞吐倍率。

随机对照使用可选命令 `v2-random-controls`，不影响本报告的主验收；吞吐、TTFT、TPOT/ITL 属于后续性能扩展。
