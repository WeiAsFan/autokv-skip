# Full 实验结果

本目录保存修复后 AutoKV-Skip full 实验的交付归档。

- Run ID：`8181c9a332ef6e9c`
- Profile：`full`
- Git commit：`2c6a4e1ac96e9835a6af2ad450cb9dfa269e4953`
- 质量：11 个配置，每配置 45 个样本；最终质量门禁通过。
- 性能：BF16、FP8、Auto-4 各 18 个场景，每个场景 3 次重复。
- 归档：`autokv-fp8-full-8181c9a332ef6e9c.tar.gz`
- 校验：`autokv-fp8-full-8181c9a332ef6e9c.tar.gz.sha256`

归档内包含：

- `runs/8181c9a332ef6e9c/`：完整实验结果、逐场景 perf 数据、报告和 manifest；
- `runs/_bugfix/20260901T013453Z-fp8-full/`：full 实验过程诊断证据。

本机校验命令：

```bash
cd /home/10360519/Files/Doc/8.26/results/full-20260901
sha256sum -c autokv-fp8-full-8181c9a332ef6e9c.tar.gz.sha256
```

