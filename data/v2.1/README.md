# v2.1 数据

正式数据在服务器首次执行 `v2-run` 时，使用已有模型的 tokenizer 离线生成到本目录下的 `quality/`。共 45 条样本，calibration 27 条、held-out 18 条；变量任务使用新的变量与值配对评分。

程序复用 `data/v2/quality/` 中已确定的 12 条自然 QA，不重新下载或挑选 LongBench。原 v2.0 数据保留不变。生成后的新数据及 manifest 会随运行输入副本和结果一起归档。

当前仓库不把开发测试使用的简化 tokenizer 数据当作正式数据。完整操作见 [v2.1 运行手册](../../docs/v2.1/RUNBOOK.zh-CN.md)。
