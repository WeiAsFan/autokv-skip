# v4.1 验证记录

2026-09-15，本地 Windows/Python 3.11，PyTorch 2.14.0 CPU。测试依赖放在忽略目录 `.cache/v41-test-deps`，没有写入项目运行依赖。

- 原有 206 项测试与新增 6 项测试，共 212 项通过。
- 数值测试覆盖 E2M1 全码本、最近偶数舍入、每组 scale、非 1 全局 scale、零值和跨页/负 slot 写入。缓存字节数包括 scale，没有 BF16 历史副本。
- FP4 完整流程测试使用生产评估器与缓存逻辑，仅替换 GPU 服务和 tokenizer，走完构造、确认、束搜索、测试和回传归档；它不是模型质量实测。
- 已对上游 c18d29d36 的真实 FlashInfer 后端应用补丁，并通过 Python 语法检查；该检查不证明 CUDA 内核可运行。
- 初次运行旧测试遇到 Windows `ADMINI~1` 与完整用户名路径不一致；统一测试进程的临时目录规范路径后全部通过，未为此改动实验代码。

复现 CPU 测试（已有 PyTorch 的 Python 环境）：

```bash
python -m unittest discover -s tests -q
```

`tests/test_v41_fp4.py` 的三项张量数值测试需要 PyTorch；无 PyTorch 时会跳过，不能将跳过称为数值验证通过。

尚未验证：RTX A6000、CUDA 12.9 上的新 FlashInfer JIT 编译，完整 Mistral 模型推理，实际混合层缓存分配，GPU 质量和容量。因此目前只能交付实现及服务器验证入口，不能声称 v4.1 GPU 实验已完成或主目标已达标。
