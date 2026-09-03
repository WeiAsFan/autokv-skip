# AutoKV-Skip v2.0 阶段 2–4 远程执行手册

状态：代码已实现，待目标 Linux 服务器生成正式数据并运行 GPU 实验

适用分支：`v2.0`

本手册只执行 v2.0 的 Quality v2 数据冻结、端点质量缺口判断、条件式层搜索和 held-out 验证。阶段 5 性能实验不在本手册范围内。

## 1. 本次运行会做什么

正式命令只有一条：

```bash
python3 -m autokv v2-run --project-root "$AUTOKV_ROOT" --port 8000 --json
```

它内部按固定顺序执行：

1. 运行 `P32` 与 `P0` 的 27 条 calibration；
2. 若 `P0` 满足冻结质量约束，不搜索层，直接运行两个端点的 18 条 held-out；
3. 若存在质量缺口，依次运行 8 个四层组、前两组中的 8 个单层、`P2 → P4 → P8`；第一个合格预算立即早停；
4. 中间策略入选时，held-out 只运行 `P32`、`P0`、所选策略和 3 个同预算随机策略；
5. 输出 `selection.json`、中文质量报告和最终 `completed-manifest.json`。

它不会运行 v1.0 的 doctor/lock/dry-run/双 smoke 链，也不会为了得到中间混合策略而绕过端点判断。

## 2. 登录并更新服务器工作区

从用于 SSH 的 Linux 设备登录小型服务器：

```bash
ssh 用户名@服务器地址
```

进入 v1.0 实际使用过的项目目录：

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip
export AUTOKV_ROOT="$(pwd -P)"
printf '项目目录：%s\n' "$AUTOKV_ROOT"
```

确认没有另一个实验进程，再更新 `v2.0`：

```bash
ps -ef | grep -E '[v]llm|[a]utokv'
git status --short --branch
git fetch --prune origin
git switch v2.0
git pull --ff-only origin v2.0
git status --short --branch
git log -1 --oneline --decorate
```

如果 `git status --short` 显示服务器自己的未提交文件，先判断归属并保留，不要使用 `git reset --hard` 或删除整个 `runs/`。

## 3. 验证代码，不启动 GPU

```bash
python3 --version
python3 scripts/verify.py
```

成功标志为最后一行：

```text
VERIFICATION_OK tests compileall quick=18 full=34 safety=passed
```

读取 v1.0 已生成的本地 vLLM 环境锁，并确认关键文件仍存在：

```bash
export AUTOKV_VLLM_PYTHON="$(python3 -c 'import json; print(json.load(open("runs/_environment/lock.json", encoding="utf-8"))["python"])')"
export AUTOKV_VLLM_BIN="$(python3 -c 'import json; print(json.load(open("runs/_environment/lock.json", encoding="utf-8"))["vllm"])')"
export AUTOKV_MODEL_PATH="$(python3 -c 'import json; print(json.load(open("runs/_environment/lock.json", encoding="utf-8"))["model_path"])')"
test -x "$AUTOKV_VLLM_PYTHON"
test -x "$AUTOKV_VLLM_BIN"
test -d "$AUTOKV_MODEL_PATH"
"$AUTOKV_VLLM_BIN" serve --help=all | grep -F -- '--no-enable-prefix-caching'
```

最后一条必须找到 `--no-enable-prefix-caching`。不要升级驱动、CUDA、torch、vLLM 或 FlashInfer 来“顺便修环境”。

## 4. 一次性导出固定 revision 的 LongBench-E 源数据

正式 GPU 运行不联网。先在一个可访问 Hugging Face 的环境中导出固定 revision 的 `qasper_e` 与 `hotpotqa_e`。服务器可联网时直接执行：

```bash
python3 -m venv .venv-data
.venv-data/bin/python -m pip install --upgrade pip
.venv-data/bin/python -m pip install 'datasets==4.0.0' 'pyarrow==21.0.0'
.venv-data/bin/python scripts/export_longbench_v2.py \
  --output-dir data/v2/source/LongBench
python3 -m json.tool data/v2/source/LongBench/source-manifest.json
```

导出器固定使用：

- 仓库：`THUDM/LongBench`；
- revision：`92b6c5fbfb0c97b91e92d9ef79802f95ce74b05e`；
- split：`test`；
- 子集：`qasper_e`、`hotpotqa_e`。

`data/v2/source/` 已被 Git 忽略，不要把完整源数据加入本项目仓库。

若小型服务器不能访问 Hugging Face，就在可联网的 Linux 设备上对同一 `v2.0` 提交执行上述导出命令，再传输三个文件：

```bash
scp data/v2/source/LongBench/source-manifest.json \
    data/v2/source/LongBench/qasper_e.jsonl \
    data/v2/source/LongBench/hotpotqa_e.jsonl \
    用户名@服务器地址:/mnt_d/huangxiaoyuan/autokv-skip/data/v2/source/LongBench/
```

服务器端再次执行：

```bash
cd /mnt_d/huangxiaoyuan/autokv-skip
python3 -m json.tool data/v2/source/LongBench/source-manifest.json
```

后续冻结命令会自动校验两个 JSONL 的 SHA-256，不接受手工替换文件。

## 5. 可选但建议：运行一次 BF16-only 难度 pilot

v1.0 出现过质量天花板，因此建议在正式数据冻结前执行一次 pilot。它只启动一次 `P32`，使用 3 个任务族 × 3 个长度 × seed 41，共 9 个请求；绝不运行 `P0`。

```bash
mkdir -p runs
"$AUTOKV_VLLM_PYTHON" -m autokv v2-pilot \
  --project-root "$AUTOKV_ROOT" \
  --port 8000 \
  --json | tee runs/v2-pilot-cli.json
python3 -m json.tool runs/v2-pilot-cli.json
```

程序按预注册规则输出 `recommended_difficulty`：

- 任一 Hard 任务族的 BF16 均值 `< 0.60`：建议 `easy`；
- 三个任务族的 BF16 均值都 `>= 0.98`：建议 `hard`；
- 其他情况：保持 `standard`。

若输出 `easy` 或 `hard`，只把 [quality.json](../../configs/v2/quality.json) 中 `data.hard.difficulty` 从 `standard` 改成对应值。不要改 seed、长度、任务参数、阈值或评分器，也不要第二次运行 pilot。若选择跳过 pilot，则保持 `standard`。

一旦 `data/v2/quality/dataset-manifest.json` 存在，`v2-pilot` 会拒绝运行，防止看见正式结果后调数据。

## 6. 冻结 45 条正式 Quality v2 数据

必须使用环境锁中的 vLLM Python，因为它包含与正式模型一致的 `transformers`、tokenizer 和 chat template：

```bash
"$AUTOKV_VLLM_PYTHON" -m autokv v2-freeze-data \
  --project-root "$AUTOKV_ROOT" \
  --source-dir data/v2/source/LongBench \
  --json | tee runs/v2-freeze-data-cli.json
python3 -m json.tool runs/v2-freeze-data-cli.json
```

检查规模与身份：

```bash
wc -l data/v2/quality/calibration.jsonl data/v2/quality/heldout.jsonl
python3 -m json.tool data/v2/quality/dataset-manifest.json
```

预期行数严格为：

```text
27 data/v2/quality/calibration.jsonl
18 data/v2/quality/heldout.jsonl
45 total
```

生成器已经执行以下检查：

- Easy 为 3 条 calibration + 3 条 held-out；
- Hard 为 18 + 9，完整覆盖 3 任务 × 3 长度 × 3 seed；
- Natural 为 6 + 6，两个数据集各 6 条；
- 合成输入经真实 chat template 后距离 8192/16384/24576 不超过 32 tokens；
- Natural 经同一 tokenizer 过滤后不超过 24576 tokens；
- 两个 split 不共享 sample ID、内容、LongBench `_id`、context 或 question。

同一有效 manifest 再次运行 `v2-freeze-data` 只会复用，不会覆盖。正式数据冻结后，不得因为 `P0` 结果不理想而重建。

## 7. 在任何 P0 结果产生前提交冻结数据

```bash
git status --short
git add configs/v2/quality.json \
        data/v2/quality/calibration.jsonl \
        data/v2/quality/heldout.jsonl \
        data/v2/quality/dataset-manifest.json
git commit -m 'data: 冻结 v2.0 质量数据'
git push origin v2.0
git status --short
git ls-files data/v2/quality
```

`git status --short` 必须为空。这样正式结果中的 Git commit、源码树 hash、配置 hash 和数据 hash 才能形成闭环。

## 8. 在 tmux 中执行正式阶段 3–4

```bash
tmux new -s autokv-v2
cd /mnt_d/huangxiaoyuan/autokv-skip
export AUTOKV_ROOT="$(pwd -P)"
python3 -m autokv v2-run \
  --project-root "$AUTOKV_ROOT" \
  --port 8000 \
  --json | tee runs/v2-run-cli.json
```

按 `Ctrl-b`、再按 `d` 脱离 tmux。重新连接 SSH 后查看：

```bash
tmux attach -t autokv-v2
```

不要同时启动第二个 `v2-run`。若 SSH 或当前命令中断，确认旧 vLLM 进程已经退出后，在同一提交、同一数据、同一端口重新执行完全相同的命令。已经具备有效 policy manifest 的策略会跳过；不完整策略会被移到对应运行目录的 `_incomplete/` 后单独重跑。

## 9. 正确理解运行规模

下列“请求数”指正式样本请求；一次暂时性 HTTP 故障最多允许在同一策略内重试一次。

| 分支 | 正式启动数 | 正式样本请求数 |
|---|---:|---:|
| 无质量缺口 | 4 | 90 |
| 有缺口且 P2 通过 | 25 | 621 |
| 有缺口且 P4 通过 | 26 | 648 |
| 有缺口且 P8 通过 | 27 | 675 |
| P8 仍失败并回退 P32 | 23 | 603 |

pilot 若执行，另加 1 次启动和 9 个 BF16 请求。程序不会临时加入 P12/P16、更多随机组或更长上下文。

## 10. 检查最终结果

正式命令成功后：

```bash
python3 -m json.tool runs/v2-run-cli.json
export AUTOKV_RUN_ID="$(python3 -c 'import json; print(json.load(open("runs/v2-run-cli.json", encoding="utf-8"))["run_id"])')"
test -f "runs/$AUTOKV_RUN_ID/decision.json"
test -f "runs/$AUTOKV_RUN_ID/selection.json"
test -f "runs/$AUTOKV_RUN_ID/report/QUALITY-v2.zh-CN.md"
test -f "runs/$AUTOKV_RUN_ID/completed-manifest.json"
python3 -m json.tool "runs/$AUTOKV_RUN_ID/decision.json"
python3 -m json.tool "runs/$AUTOKV_RUN_ID/selection.json"
sed -n '1,240p' "runs/$AUTOKV_RUN_ID/report/QUALITY-v2.zh-CN.md"
```

用最终清单校验所有活动产物：

```bash
python3 - "$AUTOKV_RUN_ID" <<'PY'
import hashlib
import json
import pathlib
import sys

run_id = sys.argv[1]
run_root = pathlib.Path("runs") / run_id
manifest = json.loads((run_root / "completed-manifest.json").read_text(encoding="utf-8"))
assert manifest["complete"] is True
assert manifest["run_id"] == run_id
for record in manifest["artifacts"]:
    path = (run_root / record["path"]).resolve()
    path.relative_to(run_root.resolve())
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == record["sha256"], (path, observed, record["sha256"])
print("V2_COMPLETED_MANIFEST_OK", len(manifest["artifacts"]))
PY
```

还应抽查每个 server 日志都证明缓存关闭：

```bash
grep -R --include='*.server.log' -L 'enable_prefix_caching=False' \
  "runs/$AUTOKV_RUN_ID/quality" || true
```

正常情况下该命令不输出任何 server log 路径。

## 11. 四种合法结论

- `final.k = 0`：P0 在 calibration 与 held-out 都满足质量约束；自动停止是正确结果，不需要制造混合策略。
- `final.k = 2/4/8` 且 `layer_selection_supported=true`：预算与层排序都得到本次证据支持。
- `final.k = 2/4/8` 且 `layer_selection_supported=false`：混合预算得到支持，但不能声称层排序优于同预算随机选择。
- `final.k = 32`：中间预算不合格或 calibration 候选未泛化，系统按设计安全回退 BF16。

任何一种都不等于阶段 5 的性能结论。当前报告不能声称吞吐、TTFT、TPOT 或 ITL 更优。

## 12. 归档并上传结果

```bash
mkdir -p "results/v2-$AUTOKV_RUN_ID"
tar -czf "results/v2-$AUTOKV_RUN_ID/autokv-v2-$AUTOKV_RUN_ID.tar.gz" \
  "runs/$AUTOKV_RUN_ID" \
  data/v2/quality \
  configs/v2/quality.json
sha256sum "results/v2-$AUTOKV_RUN_ID/autokv-v2-$AUTOKV_RUN_ID.tar.gz" \
  > "results/v2-$AUTOKV_RUN_ID/autokv-v2-$AUTOKV_RUN_ID.tar.gz.sha256"
git add "results/v2-$AUTOKV_RUN_ID"
git commit -m "results: 归档 v2.0 质量运行 $AUTOKV_RUN_ID"
git push origin v2.0
```

归档不得包含模型权重、Hugging Face token、`.cache/`、完整 LongBench 源目录或其他人的服务器文件。

## 13. 常见停止条件

| 现象 | 正确处理 |
|---|---|
| `--no-enable-prefix-caching` 不存在 | 停止；当前 runtime 与 v2 设计不兼容，不要静默省略参数 |
| 日志未出现 `enable_prefix_caching=False` | 当前策略失败；保留日志并排查，不继续累计结果 |
| 服务端 `prompt_tokens` 与冻结值不同 | tokenizer/chat template 不一致；停止，不能比较策略 |
| 首条响应含替换字符或循环片段 | 当前策略立即停止；先修 runtime，不增加 smoke 次数 |
| 单策略 JSONL 或 manifest 损坏 | 重跑同一正式命令；只重跑该策略 |
| `git status` 不干净 | 在 P0 正式运行前提交预期配置/数据，或保留并处理意外改动 |
| held-out 失败 | 输出 P32，禁止换样本或根据 held-out 重排层 |
| 达到对应资源上限 | 停止并报告，不扩大候选预算 |
