# AutoKV-Skip Implementation Plan

> **文档状态：v1.0 运行前历史实现计划。** 本文中的任务、命令、环境假设和门禁用于记录原始实施路径，不是运行 `8181c9a332ef6e9c` 的精确源码或复现说明。实际事实见 [v1.0 统一项目事实](../../v1.0/FACTS.zh-CN.md)，对应源码必须按 [v1.0 源码发布要求](../../v1.0/SOURCE-PUBLICATION-REQUIREMENT.zh-CN.md) 单独发布。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standard-library Python controller that safely runs, resumes, evaluates, and reports the approved AutoKV-Skip experiment on one driver-locked RTX A6000.

**Architecture:** A host-side CLI builds argument-vector Docker commands, starts one immutable vLLM image at a time, calls its OpenAI-compatible HTTP API, writes atomic JSON/JSONL artifacts, ranks layer sensitivity, and renders a Chinese Markdown/SVG report. GPU software remains inside Docker; all algorithmic logic is testable without a GPU through pure functions and injected command/HTTP runners.

**Tech Stack:** Python 3.10+ standard library, `unittest`, Docker CLI, official vLLM Docker images, FlashInfer, OpenAI-compatible HTTP endpoints, Markdown and SVG.

## Global Constraints

- NVIDIA driver must remain exactly `535.230.02`; no code path may install, update, downgrade, remove, or suggest changing the driver.
- Host CUDA 12.2 must not be modified or mounted into the container.
- Every GPU container must set `VLLM_ENABLE_CUDA_COMPATIBILITY=1`.
- Primary image is `vllm/vllm-openai:v0.26.0`; fallback is `vllm/vllm-openai:v0.19.1`; one run uses one locked image digest.
- Model is `mistralai/Mistral-7B-Instruct-v0.3`, compute dtype BF16, attention backend `FLASHINFER`, maximum model length 32768.
- KV cache is BF16, FP8 E4M3, or mixed through `--kv-cache-dtype-skip-layers`; weights and ordinary activations remain unquantized.
- Fixed KV cache budget is `16G`; main mixed budget is four BF16 KV layers out of 32.
- Runtime code uses only Python's standard library; tests use `unittest`.
- Host commands are passed as argument arrays with `shell=False`; secrets are redacted before logging.
- All generated paths resolve below the project root; scripts never run broad Docker cleanup or recursive deletion.
- Current repository is a new project-only checkout and is used in place; no nested worktree is created.

---

## File Map

| File | Single responsibility |
|---|---|
| `pyproject.toml` | Package metadata and console entry point |
| `autokv/config.py` | Validated profile/model/image dataclasses and JSON loading |
| `autokv/memory.py` | KV byte and ideal-capacity formulas |
| `autokv/io.py` | Atomic JSON/JSONL, hashing, path containment, redaction |
| `autokv/selection.py` | Variants, group/layer probes, top-k and controls |
| `autokv/niah.py` | Deterministic NIAH specifications and token-length fitting |
| `autokv/scoring.py` | EM, edit distance, prompt-logprob NLL, composite quality |
| `autokv/commands.py` | Subprocess result type, safe execution, Docker argv builders |
| `autokv/doctor.py` | Read-only host/image gates and immutable image lock |
| `autokv/client.py` | Standard-library vLLM HTTP client |
| `autokv/experiment.py` | Container lifecycle, request execution, resume state |
| `autokv/benchmark.py` | Capacity log parsing and vLLM benchmark command generation |
| `autokv/report.py` | Paired bootstrap, Markdown/CSV/SVG rendering |
| `autokv/cli.py` | Subcommands and end-to-end state machine |
| `configs/quick.json` | Approved default experiment matrix |
| `configs/full.json` | Approved exhaustive experiment matrix |
| `RUNBOOK.zh-CN.md` | Numbered server commands, expected output and failure branches |
| `README.md` | Project explanation and interview narrative |
| `tests/test_*.py` | Pure unit/integration-contract tests |

---

### Task 1: Package, profile validation, and KV memory model

**Files:**
- Create: `pyproject.toml`
- Create: `autokv/__init__.py`
- Create: `autokv/__main__.py`
- Create: `autokv/config.py`
- Create: `autokv/memory.py`
- Create: `tests/test_config.py`
- Create: `tests/test_memory.py`

**Interfaces:**
- Produces: `Profile.from_dict(data: dict) -> Profile`, `load_profile(path: Path) -> Profile`, `kv_bytes_per_token(...) -> int`, `mixed_kv_bytes_per_token(...) -> int`, `ideal_capacity_gain(...) -> float`.
- Consumes: no earlier task interfaces.

- [ ] **Step 1: Write failing profile and memory tests**

```python
# tests/test_memory.py
import unittest
from autokv.memory import ideal_capacity_gain, kv_bytes_per_token, mixed_kv_bytes_per_token

class MemoryTests(unittest.TestCase):
    def test_mistral_memory_numbers(self):
        self.assertEqual(kv_bytes_per_token(32, 8, 128, 2), 131072)
        self.assertEqual(kv_bytes_per_token(32, 8, 128, 1), 65536)
        self.assertEqual(mixed_kv_bytes_per_token(32, 4, 8, 128), 73728)
        self.assertAlmostEqual(ideal_capacity_gain(32, 4), 1.7777777777777777)

    def test_rejects_out_of_range_bf16_layer_count(self):
        with self.assertRaisesRegex(ValueError, "between zero and num_layers"):
            mixed_kv_bytes_per_token(32, 33, 8, 128)
```

```python
# tests/test_config.py
import unittest
from autokv.config import Profile

class ProfileTests(unittest.TestCase):
    def test_rejects_non_locked_driver(self):
        data = Profile.default_dict("quick")
        data["hardware"]["driver"] = "550.0"
        with self.assertRaisesRegex(ValueError, "535.230.02"):
            Profile.from_dict(data)

    def test_accepts_approved_quick_profile(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        self.assertEqual(profile.model.num_layers, 32)
        self.assertEqual(profile.selection.k, 4)
        self.assertEqual(profile.images[0], "vllm/vllm-openai:v0.26.0")
```

- [ ] **Step 2: Run tests and observe import failures**

Run: `python -m unittest tests.test_memory tests.test_config -v`

Expected: both modules fail to import because `autokv.memory` and `autokv.config` do not exist.

- [ ] **Step 3: Implement the minimal validated dataclasses and formulas**

```python
# autokv/memory.py
def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    values = (num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError("KV dimensions and dtype bytes must be positive")
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

def mixed_kv_bytes_per_token(num_layers: int, bf16_layers: int, num_kv_heads: int, head_dim: int) -> int:
    if not 0 <= bf16_layers <= num_layers:
        raise ValueError("bf16_layers must be between zero and num_layers")
    return 2 * num_kv_heads * head_dim * (2 * bf16_layers + num_layers - bf16_layers)

def ideal_capacity_gain(num_layers: int, bf16_layers: int) -> float:
    if num_layers <= 0 or not 0 <= bf16_layers <= num_layers:
        raise ValueError("bf16_layers must be between zero and num_layers")
    return (2 * num_layers) / (num_layers + bf16_layers)
```

`autokv/config.py` will define frozen `Hardware`, `Model`, `Selection`, `Quality`, `Benchmark`, and `Profile` dataclasses. `Profile.from_dict` will require the exact driver, image order, model ID, layer count, backend, dtype, 16G KV budget, and `k=4`; `default_dict` supplies the approved values so tests and shipped JSON share one schema.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_memory tests.test_config -v`

Expected: 4 tests pass with no warnings.

- [ ] **Step 5: Commit the package foundation**

```bash
git add pyproject.toml autokv tests/test_config.py tests/test_memory.py
git commit -m "feat: add validated experiment profile"
```

---

### Task 2: Atomic artifacts, hashing, containment, and secret redaction

**Files:**
- Create: `autokv/io.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Produces: `ensure_within(root: Path, path: Path) -> Path`, `atomic_write_json(path, data)`, `append_jsonl(path, row)`, `read_json(path)`, `sha256_file(path)`, `redact(value, secrets) -> str`.
- Consumes: only Python standard library.

- [ ] **Step 1: Write failing safety tests**

```python
import tempfile
import unittest
from pathlib import Path
from autokv.io import atomic_write_json, ensure_within, read_json, redact, sha256_file

class IoTests(unittest.TestCase):
    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "outside project root"):
                ensure_within(root, root.parent / "escaped.json")

    def test_atomic_json_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(read_json(path), {"a": 1, "b": 2})
            self.assertEqual(len(sha256_file(path)), 64)

    def test_redacts_exact_secret(self):
        self.assertEqual(redact("Authorization: abc123", ["abc123"]), "Authorization: ***")
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_io -v`

Expected: import failure for `autokv.io`.

- [ ] **Step 3: Implement atomic JSON and redaction**

```python
def ensure_within(root: Path, path: Path) -> Path:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise ValueError(f"path is outside project root: {path_resolved}")
    return path_resolved

def redact(value: str, secrets: Iterable[str]) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "***")
    return result
```

`atomic_write_json` will create a same-directory temporary file with `mkstemp`, serialize sorted UTF-8 JSON, flush and `os.fsync`, then `os.replace`. `append_jsonl` will serialize one compact row, flush and fsync. No function deletes data.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_io -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit artifact safety**

```bash
git add autokv/io.py tests/test_io.py
git commit -m "feat: add safe artifact persistence"
```

---

### Task 3: Layer variants, coarse-to-fine selection, and controls

**Files:**
- Create: `autokv/selection.py`
- Create: `tests/test_selection.py`

**Interfaces:**
- Produces: frozen `Variant`, `group_layers`, `group_probe_variants`, `layer_probe_variants`, `select_top_groups`, `select_top_layers`, `random_controls`, `canonical_config_id`.
- Consumes: `Profile.selection` from Task 1 and `sha256` from Python.

- [ ] **Step 1: Write failing selection tests**

```python
import unittest
from autokv.selection import group_layers, random_controls, select_top_layers

class SelectionTests(unittest.TestCase):
    def test_groups_32_layers_into_eight_contiguous_groups(self):
        groups = group_layers(32, 4)
        self.assertEqual(groups[0], (0, 1, 2, 3))
        self.assertEqual(groups[-1], (28, 29, 30, 31))
        self.assertEqual(len(groups), 8)

    def test_selects_highest_scored_layers_with_stable_tie_break(self):
        scores = {7: 0.3, 2: 0.3, 9: -0.1, 4: 0.2}
        self.assertEqual(select_top_layers(scores, 2), (2, 7))

    def test_random_controls_are_unique_and_do_not_duplicate_named_sets(self):
        controls = random_controls(32, 4, [11, 23, 37, 53, 71], {(0, 1, 2, 3), (28, 29, 30, 31)})
        self.assertEqual(len(controls), 5)
        self.assertEqual(len(set(controls)), 5)
        self.assertTrue(all(len(item) == 4 for item in controls))
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_selection -v`

Expected: import failure for `autokv.selection`.

- [ ] **Step 3: Implement deterministic selection**

```python
def select_top_layers(scores: Mapping[int, float], k: int) -> tuple[int, ...]:
    if k <= 0 or k > len(scores):
        raise ValueError("k must select at least one available layer")
    ranked = sorted(scores, key=lambda layer: (-scores[layer], layer))
    return tuple(sorted(ranked[:k]))

def group_layers(num_layers: int, group_size: int) -> tuple[tuple[int, ...], ...]:
    if num_layers <= 0 or group_size <= 0 or num_layers % group_size:
        raise ValueError("num_layers must be evenly divisible by group_size")
    return tuple(tuple(range(start, start + group_size)) for start in range(0, num_layers, group_size))
```

`Variant` will validate sorted, unique, in-range layers and expose `bf16`, `fp8`, and `mixed` constructors. `random_controls` will advance deterministically beyond the supplied seeds when a duplicate or forbidden set occurs.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_selection -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit selection logic**

```bash
git add autokv/selection.py tests/test_selection.py
git commit -m "feat: add coarse-to-fine layer selection"
```

---

### Task 4: Deterministic NIAH generation and quality scoring

**Files:**
- Create: `autokv/niah.py`
- Create: `autokv/scoring.py`
- Create: `tests/test_niah.py`
- Create: `tests/test_scoring.py`

**Interfaces:**
- Produces: `NiahCase`, `make_cases(profile)`, `fit_prompt(case, count_tokens)`, `expected_answer(code)`, `score_generation`, `answer_nll_from_echo`, `quality_score`.
- Consumes: quality lengths/depths/seeds from Task 1.

- [ ] **Step 1: Write failing NIAH tests**

```python
import unittest
from autokv.niah import NiahCase, expected_answer, fit_prompt

class NiahTests(unittest.TestCase):
    def test_expected_answer_repeats_code_five_times(self):
        self.assertEqual(expected_answer("ZEBRA-4821"), "|".join(["ZEBRA-4821"] * 5))

    def test_fit_prompt_is_deterministic_and_within_half_percent(self):
        case = NiahCase("case-1", 2000, 0.5, 42, "ZEBRA-4821")
        count_tokens = lambda text: len(text.split())
        first = fit_prompt(case, count_tokens)
        second = fit_prompt(case, count_tokens)
        self.assertEqual(first, second)
        self.assertLessEqual(abs(first.token_count - 2000), 10)
        self.assertIn("VERIFICATION-CODE: ZEBRA-4821", first.prompt)
```

- [ ] **Step 2: Write failing scoring tests and run RED**

```python
import math
import unittest
from autokv.scoring import answer_nll_from_echo, score_generation

class ScoringTests(unittest.TestCase):
    def test_generation_score_normalizes_whitespace_and_case(self):
        expected = "A-1|A-1|A-1|A-1|A-1"
        score = score_generation("  a-1 | a-1 | a-1 | a-1 | a-1  ", expected)
        self.assertEqual(score.exact_match, 1.0)

    def test_extracts_answer_nll_from_openai_echo_offsets(self):
        response = {"choices": [{"logprobs": {"token_logprobs": [None, -0.2, -0.4], "text_offset": [0, 10, 12]}}]}
        self.assertAlmostEqual(answer_nll_from_echo(response, 10), 0.3)
```

Run: `python -m unittest tests.test_niah tests.test_scoring -v`

Expected: imports fail because both production modules are absent.

- [ ] **Step 3: Implement deterministic prompts and scoring**

`fit_prompt` will binary-search the number of repetitions of a fixed public-domain-style filler sentence, insert the needle at the requested repetition depth, and require `abs(actual-target) <= max(1, round(target*0.005))`. It returns a frozen `MaterializedPrompt(prompt, token_count, needle_depth)` and raises a descriptive error after 32 iterations if the tokenizer callback is non-monotonic.

```python
def quality_score(exact_match: float, answer_nll: float | None, edit_distance: int, expected_length: int) -> float:
    if answer_nll is not None:
        probability = math.exp(-min(max(answer_nll, 0.0), 20.0))
        return 0.8 * exact_match + 0.2 * probability
    normalized = edit_distance / max(expected_length, 1)
    return exact_match - 0.001 * normalized
```

The Levenshtein implementation will use two integer rows, keeping memory linear in output length.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_niah tests.test_scoring -v`

Expected: 4 tests pass.

- [ ] **Step 5: Commit NIAH and scoring**

```bash
git add autokv/niah.py autokv/scoring.py tests/test_niah.py tests/test_scoring.py
git commit -m "feat: add deterministic long-context scoring"
```

---

### Task 5: Safe command execution and Docker argv generation

**Files:**
- Create: `autokv/commands.py`
- Create: `tests/test_commands.py`

**Interfaces:**
- Produces: `CommandResult`, `run_command`, `format_command`, `image_probe_commands`, `server_command`, `bench_command`, `container_name`.
- Consumes: `Profile` and `Variant` from Tasks 1 and 3.

- [ ] **Step 1: Write failing command-contract tests**

```python
import unittest
from pathlib import Path
from autokv.commands import server_command
from autokv.config import Profile
from autokv.selection import Variant

class CommandTests(unittest.TestCase):
    def test_mixed_server_command_contains_compatibility_and_skip_layers(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        argv = server_command(profile, "sha256:locked", Variant.mixed("auto-4", (2, 7, 18, 29)), Path("/srv/autokv"), 8000, "run1")
        joined = " ".join(argv)
        self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=1", joined)
        self.assertIn("--attention-backend FLASHINFER", joined)
        self.assertIn("--kv-cache-dtype fp8_e4m3", joined)
        self.assertIn("--kv-cache-dtype-skip-layers 2 7 18 29", joined)
        self.assertNotIn("nvidia-driver", joined.lower())

    def test_bf16_command_does_not_calculate_fp8_scales(self):
        profile = Profile.from_dict(Profile.default_dict("quick"))
        argv = server_command(profile, "sha256:locked", Variant.bf16(), Path("/srv/autokv"), 8000, "run1")
        self.assertNotIn("--calculate-kv-scales", argv)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_commands -v`

Expected: import failure for `autokv.commands`.

- [ ] **Step 3: Implement argv-only commands**

```python
def run_command(argv: Sequence[str], timeout: float | None = None, env: Mapping[str, str] | None = None) -> CommandResult:
    completed = subprocess.run(list(argv), shell=False, text=True, capture_output=True, timeout=timeout, env=None if env is None else dict(env))
    return CommandResult(tuple(argv), completed.returncode, completed.stdout, completed.stderr)
```

`server_command` will return `docker run -d --rm` with one exact container name, the project label, `--gpus all`, `--ipc=host`, `--network=host`, compatibility env, explicit Hugging Face cache mount, and the approved vLLM arguments. It will never embed a shell command, wildcard, host CUDA mount, privileged mode, or package installer.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_commands -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit command construction**

```bash
git add autokv/commands.py tests/test_commands.py
git commit -m "feat: add safe Docker command builder"
```

---

### Task 6: Read-only doctor and immutable image lock

**Files:**
- Create: `autokv/doctor.py`
- Create: `tests/test_doctor.py`

**Interfaces:**
- Produces: `HostFacts`, `Gate`, `parse_gpu_csv`, `validate_host`, `classify_pull_failure`, `parse_hf_revision`, `lock_first_compatible_image`.
- Consumes: `CommandResult/run_command` from Task 5 and atomic JSON from Task 2.

- [ ] **Step 1: Write failing doctor tests**

```python
import unittest
from autokv.doctor import classify_pull_failure, parse_gpu_csv, parse_hf_revision, validate_host

class DoctorTests(unittest.TestCase):
    def test_parses_expected_a6000(self):
        facts = parse_gpu_csv("NVIDIA RTX A6000, 535.230.02, 49140, 8.6\n")
        self.assertEqual(facts.driver, "535.230.02")
        self.assertEqual(facts.compute_capability, "8.6")

    def test_rejects_driver_change(self):
        facts = parse_gpu_csv("NVIDIA RTX A6000, 550.54.15, 49140, 8.6\n")
        gates = validate_host(facts, expected_driver="535.230.02")
        self.assertFalse(all(gate.ok for gate in gates))

    def test_network_pull_error_does_not_allow_version_fallback(self):
        self.assertEqual(classify_pull_failure("TLS handshake timeout"), "network")

    def test_reads_immutable_hugging_face_revision(self):
        self.assertEqual(parse_hf_revision({"sha": "a" * 40}), "a" * 40)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_doctor -v`

Expected: import failure for `autokv.doctor`.

- [ ] **Step 3: Implement gates and image protocol**

`lock_first_compatible_image` will execute these stages in order: host driver query, Docker daemon query, GPU container query, candidate pull, CUDA/FlashInfer import, vLLM help flag inspection, image inspect, and Hugging Face model metadata resolution. Pull failure classified as network/auth/disk stops immediately. A pulled image whose CUDA/FlashInfer/feature probe fails permits the next candidate. The successful inspect object is atomically stored with image ID, RepoDigest, container package versions, immutable 40-hex model revision and UTC timestamp. An unresolved or malformed model revision is a gate failure and is never replaced by a mutable branch name.

```python
REQUIRED_FLAGS = (
    "--attention-backend",
    "--kv-cache-dtype",
    "--kv-cache-dtype-skip-layers",
    "--kv-cache-memory-bytes",
    "--calculate-kv-scales",
)
```

Every gate record will contain `name`, `ok`, `observed`, `expected`, and `remediation`; remediation for driver mismatch is only “确认连接的是目标服务器，不要修改驱动”.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_doctor -v`

Expected: 4 tests pass.

- [ ] **Step 5: Commit doctor**

```bash
git add autokv/doctor.py tests/test_doctor.py
git commit -m "feat: add driver-locked environment doctor"
```

---

### Task 7: HTTP client, server lifecycle, execution, and resume

**Files:**
- Create: `autokv/client.py`
- Create: `autokv/experiment.py`
- Create: `tests/test_client.py`
- Create: `tests/test_experiment.py`

**Interfaces:**
- Produces: `VllmClient`, `wait_until_ready`, `ExperimentRunner`, `canonical_run_id`, `mark_complete`, `is_complete`.
- Consumes: Tasks 1–6.

- [ ] **Step 1: Write failing client and resume tests**

```python
import unittest
from autokv.client import VllmClient

class ClientTests(unittest.TestCase):
    def test_completion_payload_is_deterministic(self):
        payload = VllmClient.completion_payload("model", "prompt", 24)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_tokens"], 24)
```

```python
import tempfile
import unittest
from pathlib import Path
from autokv.experiment import mark_complete, is_complete

class ExperimentTests(unittest.TestCase):
    def test_complete_state_requires_matching_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "rows.jsonl"
            artifact.write_text('{"ok":true}\n', encoding="utf-8")
            mark_complete(root / "state.json", artifact, expected_rows=1)
            self.assertTrue(is_complete(root / "state.json", artifact, expected_rows=1))
            artifact.write_text('{"ok":false}\n', encoding="utf-8")
            self.assertFalse(is_complete(root / "state.json", artifact, expected_rows=1))
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_client tests.test_experiment -v`

Expected: imports fail for the two missing modules.

- [ ] **Step 3: Implement client and runner state machine**

`VllmClient` will use `urllib.request` with JSON content type, explicit timeouts, and bounded response sizes. It exposes `health`, `tokenize`, `complete`, and `echo_logprobs`. HTTP errors preserve status and a truncated, secret-redacted body.

`ExperimentRunner.run_variant` will:

1. verify that image lock and dataset hash exist;
2. skip only when `is_complete` validates count and SHA-256;
3. start the exact labeled container;
4. poll `/health` with Docker log capture on failure;
5. assert logs contain FlashInfer, requested dtype and capacity;
6. materialize prompts, issue completion and echo requests sequentially;
7. append one result row per case;
8. stop only its exact labeled container in `finally`;
9. atomically mark completion.

```python
def canonical_run_id(profile_hash: str, image_digest: str, model_revision: str, dataset_hash: str) -> str:
    raw = "\n".join((profile_hash, image_digest, model_revision, dataset_hash)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_client tests.test_experiment -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit experiment execution**

```bash
git add autokv/client.py autokv/experiment.py tests/test_client.py tests/test_experiment.py
git commit -m "feat: add resumable vLLM experiment runner"
```

---

### Task 8: Capacity/performance benchmark and report generation

**Files:**
- Create: `autokv/benchmark.py`
- Create: `autokv/report.py`
- Create: `tests/test_benchmark.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Produces: `parse_capacity_tokens`, `build_benchmark_matrix`, `paired_bootstrap_ci`, `render_report`, `render_summary_csv`, `render_capacity_svg`.
- Consumes: result rows, selection JSON, profile and memory formulas.

- [ ] **Step 1: Write failing log and statistics tests**

```python
import unittest
from autokv.benchmark import parse_capacity_tokens

class BenchmarkTests(unittest.TestCase):
    def test_parses_vllm_capacity_log(self):
        log = "GPU KV cache size: 233,104 tokens\nMaximum concurrency for 32,768 tokens per request: 7.11x"
        parsed = parse_capacity_tokens(log)
        self.assertEqual(parsed.tokens, 233104)
        self.assertAlmostEqual(parsed.max_concurrency, 7.11)
```

```python
import unittest
from autokv.report import paired_bootstrap_ci

class ReportTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self):
        first = paired_bootstrap_ci([1.0, 0.8, 0.9], [0.7, 0.6, 0.8], seed=42, samples=1000)
        second = paired_bootstrap_ci([1.0, 0.8, 0.9], [0.7, 0.6, 0.8], seed=42, samples=1000)
        self.assertEqual(first, second)
        self.assertGreater(first.mean, 0)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_benchmark tests.test_report -v`

Expected: imports fail for both production modules.

- [ ] **Step 3: Implement parsing and deterministic rendering**

`parse_capacity_tokens` will support comma-separated integers and both “GPU KV cache size” and “Maximum concurrency” lines; missing capacity is a hard error. `paired_bootstrap_ci` uses `random.Random(seed)` and resamples paired indices, returning mean, 2.5th and 97.5th percentiles.

`render_report` will write:

- environment and immutable versions;
- selected layers and sensitivity table;
- theoretical versus measured capacity;
- EM, Q, answer NLL and paired confidence intervals;
- BF16/FP8/Auto performance table;
- pre-registered success checks;
- explicit null/negative-result wording;
- limitations of random-token scales and Ampere FP8.

`render_capacity_svg` will emit plain SVG rectangles and labels without third-party plotting libraries.
`render_summary_csv` will use `csv.DictWriter` with a fixed column order so the same artifact can be opened directly in a spreadsheet. `run_benchmarks` will start BF16, FP8 and Auto-4 servers one at a time, invoke the image-matched `vllm bench serve` client command for 1K/8K/16K input scenarios, preserve raw JSON, and stop only the labeled server in a `finally` block.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_benchmark tests.test_report -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit benchmarking and reporting**

```bash
git add autokv/benchmark.py autokv/report.py tests/test_benchmark.py tests/test_report.py
git commit -m "feat: add capacity benchmark and report"
```

---

### Task 9: CLI, profiles, dry-run, diagnostics, and end-to-end orchestration

**Files:**
- Create: `autokv/cli.py`
- Create: `configs/quick.json`
- Create: `configs/full.json`
- Create: `tests/test_cli.py`
- Modify: `autokv/__main__.py`

**Interfaces:**
- Produces commands: `doctor`, `lock-image`, `make-data`, `dry-run`, `smoke`, `probe`, `select`, `evaluate`, `benchmark`, `report`, `status`, `diagnose`, `run`.
- Consumes: all prior tasks.

- [ ] **Step 1: Write failing CLI contract tests**

```python
import json
import subprocess
import sys
import unittest

class CliTests(unittest.TestCase):
    def test_help_lists_complete_server_workflow(self):
        result = subprocess.run([sys.executable, "-m", "autokv", "--help"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        for command in ("doctor", "dry-run", "smoke", "probe", "evaluate", "benchmark", "report", "run", "status", "diagnose"):
            self.assertIn(command, result.stdout)

    def test_quick_dry_run_is_json_and_never_invokes_docker(self):
        result = subprocess.run([sys.executable, "-m", "autokv", "dry-run", "--profile", "quick", "--json"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["profile"], "quick")
        self.assertEqual(data["core_probe_configurations"], 18)
        self.assertFalse(data["executed"])
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_cli -v`

Expected: help is incomplete and dry-run exits nonzero.

- [ ] **Step 3: Implement argparse CLI and workflow state machine**

All commands accept `--project-root` defaulting to the repository root and `--profile quick|full`. `doctor` and real GPU commands require Linux; `dry-run`, `make-data`, `select`, `report`, and tests remain cross-platform.

`run --profile quick` executes this exact state sequence with state validation between transitions:

```python
RUN_STEPS = (
    "doctor",
    "lock-image",
    "make-data",
    "smoke",
    "probe",
    "select",
    "evaluate",
    "benchmark",
    "report",
)
```

`smoke` starts the same full-FP8 configuration twice from clean engines, sends one fixed 1024-token NIAH request each time, and compares output text, output token count and parsed KV capacity. Any mismatch exits with code 5 and prints the exact dataset-calibration extension command; it does not continue to `probe`. `probe` runs all eight group variants, chooses two groups, runs their eight layer variants, and measures the joint Auto-4. `evaluate` materializes BF16, FP8, Auto-4, five unique Random-4, First-4, Last-4 and Inverted-4 against the same final dataset hash.

`status` reads artifacts only. `diagnose` creates a gzip tar containing redacted logs, configs, environment facts, state JSON and hashes; it excludes HF cache, model files and tokens. Exit codes are `0` success, `2` invalid input/gate, `3` external command, `4` HTTP/server, `5` incomplete data.

- [ ] **Step 4: Verify GREEN and full suite**

Run: `python -m unittest tests.test_cli -v`

Expected: 2 tests pass.

Run: `python -m unittest discover -s tests -v`

Expected: every test from Tasks 1–9 passes.

- [ ] **Step 5: Commit CLI and profiles**

```bash
git add autokv/cli.py autokv/__main__.py configs tests/test_cli.py
git commit -m "feat: add end-to-end AutoKV-Skip CLI"
```

---

### Task 10: Chinese runbook, research summary, and interview guide

**Files:**
- Create: `README.md`
- Create: `RUNBOOK.zh-CN.md`
- Create: `docs/research/inference-optimization-landscape.zh-CN.md`
- Create: `docs/interview/AutoKV-Skip-interview-guide.zh-CN.md`
- Create: `tests/test_docs.py`

**Interfaces:**
- Produces: standalone operator and interview documentation.
- Consumes: exact CLI commands and paths from Task 9.

- [ ] **Step 1: Write failing documentation completeness test**

```python
import unittest
from pathlib import Path

class DocumentationTests(unittest.TestCase):
    def test_runbook_has_every_numbered_phase_and_no_unfinished_marker(self):
        text = Path("RUNBOOK.zh-CN.md").read_text(encoding="utf-8")
        for phase in range(0, 14):
            self.assertIn(f"阶段 {phase}", text)
        for marker in ("T" + "BD", "T" + "ODO", "CHANGE" + "ME"):
            self.assertNotIn(marker, text)
        self.assertIn("535.230.02", text)
        self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=1", text)
        self.assertIn("不要升级驱动", text)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_docs -v`

Expected: `RUNBOOK.zh-CN.md` is missing.

- [ ] **Step 3: Write exact, self-contained documentation**

The runbook phases are fixed:

0. print/copy this checklist;
1. identify Linux distribution and create the project-local environment variables;
2. read-only host inspection;
3. Docker and NVIDIA Container Toolkit gate, with separate Ubuntu/Debian and RHEL/Rocky repair appendices that never modify the driver;
4. optional HF token input without echo;
5. `doctor` and expected JSON fields;
6. `lock-image` and digest verification;
7. model cache warmup and free-space check;
8. `make-data` plus `dry-run`;
9. 10-minute smoke and two-run scale determinism gate;
10. detached `tmux` quick run;
11. status, log reading, SSH reconnect and resume;
12. report acceptance checklist and export;
13. optional full profile and exact conditions for running it.

Every command block will be directly copyable and followed by expected success text, maximum wait guidance, and a numbered failure branch. The research document covers FlashAttention, PagedAttention, FlashInfer, quantization, compression and fusion with authoritative source links. The interview guide contains 30-second, 5-minute and 15-minute narratives, formulas, expected objections, null-result wording and extension ideas.

- [ ] **Step 4: Verify documentation and CLI examples**

Run: `python -m unittest tests.test_docs -v`

Expected: 1 test passes.

Run: `python -m autokv dry-run --profile quick --json`

Expected: valid JSON with `core_probe_configurations` equal to 18 and `executed` false.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md RUNBOOK.zh-CN.md docs/research docs/interview tests/test_docs.py
git commit -m "docs: add standalone server runbook"
```

---

### Task 11: Static safety audit and release verification

**Files:**
- Create: `tests/test_safety.py`
- Create: `scripts/verify.py`
- Modify: `README.md`

**Interfaces:**
- Produces: one cross-platform verification command, `python scripts/verify.py`.
- Consumes: entire repository.

- [ ] **Step 1: Write safety tests that require the not-yet-created verifier**

```python
import unittest
from pathlib import Path
from scripts.verify import verification_commands

FORBIDDEN_RUNTIME_PATTERNS = (
    "apt install nvidia-driver",
    "dnf install nvidia-driver",
    "docker system prune",
    "--privileged",
    "/usr/local/cuda:/usr/local/cuda",
)

class SafetyTests(unittest.TestCase):
    def test_runtime_python_contains_no_forbidden_mutation(self):
        text = "\n".join(path.read_text(encoding="utf-8") for path in Path("autokv").glob("*.py"))
        for pattern in FORBIDDEN_RUNTIME_PATTERNS:
            self.assertNotIn(pattern, text.lower())

    def test_release_verifier_runs_tests_compile_and_both_dry_runs(self):
        commands = verification_commands()
        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("unittest discover" in command for command in flattened))
        self.assertTrue(any("compileall" in command for command in flattened))
        self.assertTrue(any("dry-run --profile quick" in command for command in flattened))
        self.assertTrue(any("dry-run --profile full" in command for command in flattened))
```

- [ ] **Step 2: Verify RED before the verifier exists**

Run: `python -m unittest tests.test_safety -v`

Expected: import failure for `scripts.verify`, proving the verifier contract is not implemented.

- [ ] **Step 3: Implement the release verifier**

`scripts/verify.py` will expose `verification_commands()` returning the following tuples and run them in order using `subprocess.run(shell=False)`:

```python
COMMANDS = (
    (sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"),
    (sys.executable, "-m", "compileall", "-q", "autokv", "scripts"),
    (sys.executable, "-m", "autokv", "dry-run", "--profile", "quick", "--json"),
    (sys.executable, "-m", "autokv", "dry-run", "--profile", "full", "--json"),
)
```

It exits at the first nonzero command, prints the exact argv, and finally validates that both dry-run outputs are JSON and no generated command includes a forbidden runtime pattern. README will name this as the release verification command.

- [ ] **Step 4: Run fresh full verification**

Run: `python scripts/verify.py`

Expected: all unit tests pass, compileall exits zero, quick dry-run reports 18 core configurations, full dry-run reports 34 core configurations, and safety scan passes.

Run: `git diff --check`

Expected: no output and exit code zero.

- [ ] **Step 5: Commit verification and inspect the final tree**

```bash
git add scripts/verify.py tests/test_safety.py README.md
git commit -m "test: add release safety verification"
git status --short --branch
git log --oneline --decorate -12
```

Expected: clean `main` branch with the design, plan, implementation and verification commits visible.

---

## Plan Self-Review Record

- Spec coverage: tasks cover driver immutability, image compatibility/locking, model/backend/KV settings, memory formulas, quick/full selection, controls, NIAH/NLL scoring, capacity/performance, resume, diagnostics, report, runbook and interview material.
- Scope boundary: dataset-calibrated LLM Compressor scales remain a documented extension, as approved; no code task quantizes weights or ordinary activations.
- Type consistency: `Profile`, `Variant`, `CommandResult`, `VllmClient`, `ExperimentRunner`, result JSONL and report inputs flow in one direction with no circular module imports.
- GPU boundary: all tests before server access are pure or use injected runners; the runbook labels target-server gates separately.
- Execution mode: inline execution with `superpowers:executing-plans`, because current collaboration rules do not permit unsolicited subagent dispatch.
