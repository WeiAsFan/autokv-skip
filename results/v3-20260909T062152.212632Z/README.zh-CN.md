# AutoKV-Skip v3 运行归档

导出时间（UTC）：20260909T062152.212632Z

包含原始回答、部分结果、每次服务日志、搜索轨迹、运行输入与 CLI 失败日志。
`runs/<run_id>/inputs/` 为运行时实际输入；根目录源码是导出时副本。
来源和许可说明位于 inputs 的 dataset-manifest.json；自定义提示与评分不是官方榜单协议。

| 运行 | 状态 | 主实验达标 |
|---|---|---|
| v3-a502366f75cc8179 | test_quality_failed | False |

在 Linux 登录设备浏览器选择 GitHub 的 v3.0 分支，上传本目录文件并手动提交。服务器与 Linux 设备均不需要 git push。

## 解包

```bash
cat autokv-v3.tar.gz.part-* > autokv-v3.tar.gz
tar -xzf autokv-v3.tar.gz
```
