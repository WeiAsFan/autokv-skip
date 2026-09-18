# AutoKV-Skip v4.2 质量与容量报告

运行：`v42-5ce6b6ae69fb813a`；状态：`no_feasible_within_budget`。
流程完成：`True`；主实验达标：`False`。

质量判断采用预定点估计约束：总分下降不超过 0.01、每任务不超过 0.02。
置信区间未经同时覆盖校正；点估计达标不等于已经证明 1% 非劣性。

实际容差：`{'all': 0.01, 'single_lookup': 0.02}`；容量目标：`2.0×`。
FP4 E2M1：每 16 个数动态 E4M3 scale，全局因子 1；BF16 Q/O；prefix caching 关闭。

冻结候选：`None`（层号从 0 开始）。
选择只使用 experiment；测试失败不会自动换层。P32 回退不是容量优化成功。

两批构造确认通过：`True`；测试复现数据条件：`None`。
强恢复场景成立：`False`；测试两端数据条件：`None`。
若实验集的 P0 已满足最终容差，保留早停结论，不重新抽题或强制选择 BF16 层。

直接复用 v4.1 运行 `v41-40ef467cef6a32d2` 的数据，数据集 `cdfa714f4b3cdada1efa6208`。
本次未重新构造或确认；两批确认是源运行的历史证据。

搜索到第 9 层；完整评估候选 6/6，其中提前确认 3 个。
复筛排序决定下一轮父组合；完整实验集确认后冻结，不追加反向删层。预算内未找到不代表不存在可行解。

数据题数：`{'experiment': 512, 'test': 4096}`；来源组数：`{'experiment': 512, 'test': 4096}`。
来源划分：同阶段可复用背景；区间以新映射实例为抽样单位，只解释固定背景池条件下的分布。
背景文档数：`{'experiment': 16431, 'test': 130182}`；背景复用和原始来源见输入元数据。

背景复用统计：`{'experiment': {'document_uses': 19790, 'maximum_samples_per_document': 5, 'reuse_fraction': 0.1697321879737241, 'unique_documents': 16431}, 'test': {'document_uses': 156997, 'maximum_samples_per_document': 7, 'reuse_fraction': 0.17079944202755468, 'unique_documents': 130182}}`。

## experiment 分数

| 配置 | 总分 | single_lookup |
|---|---:|---:|
| P32 | 0.882812 | 0.882812 |
| P0 | 0.757812 | 0.757812 |

| 配置 | 任务与长度 | 分数 |
|---|---|---:|
| P32 | single_lookup:8192 | 0.882812 |
| P0 | single_lookup:8192 | 0.757812 |

## 计算开销

| 阶段 | 请求 | 重试 | 服务启动 | 墙钟秒 |
|---|---:|---:|---:|---:|
| development | 0 | 0 | 0 | 0.000 |
| endpoints | 1024 | 0 | 2 | 1478.350 |
| search | 24128 | 0 | 545 | 53581.338 |
| test | 0 | 0 | 0 | 0.000 |

合计：`{'requests': 25152, 'retries': 0, 'server_starts': 547, 'seconds': 55059.68764203088}`。
样本与配置统计：`{'full_experiment_configs': 8, 'partial_experiment_configs': 460, 'successful_answers': 25152, 'input_tokens': 206628552, 'output_tokens': 201259, 'length_finished': 0, 'empty_answers': 0}`。

存在未写完结束时间的启动记录：`False`；若为 true，墙钟统计只是已记录的下界。

仅能解释本次预先确定的任务分布。有限束宽与早期淘汰不保证全局最优；固定题数不自动保证统计检验能力。
未执行方法对照或吞吐实验时，不声称搜索优于其他算法或推理更快。
