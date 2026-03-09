# Test Notes

Notes on key test files, what they cover, and observed results.

---

## `tests/parallel/ulysses/test_async_ulysses.py`

**Purpose:** Verifies that async Ulysses sequence parallelism (SP) produces numerically identical results to standard data parallelism for both forward outputs and gradients.

**Requirements:** ≥4 GPUs; skipped on NPU.

**Run:**
```bash
pytest tests/parallel/ulysses/test_async_ulysses.py -v
# 2 passed in ~22s
```

### `test_self_attn` — aligned sequence length

- Input shape: `[2, 8192, 1024]` (seq_len=8192, divisible by SP degree)
- Runs attention in two modes with identical weights:
  - **Async SP**: input sharded across ranks via `slice_input_tensor`
  - **Plain DP**: full input on each rank
- Loss: `output.sum() * 2` (overlapping-friendly gradient)
- Assertions (all pass): forward outputs, weight grads (`proj_o`, `q_proj`), and input grads match within tight tolerances (`atol≤1e-4`, `rtol≤1e-4`)

### `test_self_attn_padding` — non-aligned sequence length

- Input shape: `[2, 8191, 1024]` (seq_len=8191, **not** divisible by SP degree)
- Exercises the padding path: SP pads to next multiple, computes, then unpads after gather
- Loss: `sum(output * ones)` (non-overlapping gradient)
- Same set of numerical assertions as above

---

## `tests/e2e/test_e2e_parallel.py`

**Purpose:** End-to-end parallel alignment tests. Verifies that training under different parallelism configurations (FSDP, varying SP degrees) produces consistent loss values across runs.

**Requirements:** transformers v4 (most cases); one case requires v5 and is skipped with v4.

**Run:**
```bash
pytest tests/e2e/test_e2e_parallel.py -v
# 12 passed, 1 skipped in ~18 min
```

**How each test works:**
1. Initializes random weights from a toy model config
2. Launches training (`train_text` or `train_vlm`) under multiple SP/parallelism configs as subprocesses
3. Loads the logged loss dicts from each run
4. Asserts all runs agree within `rtol=0.1`, `atol=0.1`

### `test_text_parallel_align` — text/LLM models

| Model | Config | MoE | Status |
|---|---|---|---|
| llama3.1 | `tests/toy_config/llama31_toy` | No | PASSED |
| qwen2.5 | `tests/toy_config/qwen25_toy` | No | PASSED |
| qwen3 | `tests/toy_config/qwen3_toy` | No | PASSED |
| qwen3_moe | `tests/toy_config/qwen3_moe_toy` | Yes | PASSED |
| seed_oss | `tests/toy_config/seed_oss_toy` | No | PASSED |
| deepseek_v3 | `tests/toy_config/deepseek_v3_toy` | Yes | PASSED |
| qwen3_5 | `tests/toy_config/qwen3_5_toy/config.json` | No | SKIPPED (requires transformers ≥ 5.0.0) |

### `test_qwen2vl_parallel_align` — Qwen2-VL models

| Model | Config | MoE | Status |
|---|---|---|---|
| qwen2vl | `tests/toy_config/qwen2vl_toy` | No | PASSED |
| qwen25vl | `tests/toy_config/qwen25vl_toy` | No | PASSED |

### `test_qwen3vl_parallel_align` — Qwen3-VL models

| Model | Config | MoE | Status |
|---|---|---|---|
| qwen3vl | `tests/toy_config/qwen3vl_toy` | No | PASSED |
| qwen3vlmoe | `tests/toy_config/qwen3vlmoe_toy` | Yes | PASSED |

### `test_qwen2omni_parallel_align` — Qwen2-Omni models

| Model | Config | MoE | Status |
|---|---|---|---|
| qwen25_omni | `tests/toy_config/qwen25omni_toy` | No | PASSED |

### `test_qwen3omni_parallel_align` — Qwen3-Omni models

| Model | Config | MoE | Status |
|---|---|---|---|
| qwen3_omni_moe | `tests/toy_config/qwen3omni_toy` | Yes | PASSED |
