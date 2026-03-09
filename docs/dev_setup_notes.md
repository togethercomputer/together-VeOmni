# VeOmni Dev Setup Notes

## Environment Setup

### 1. uv Version Pinning

`pyproject.toml` pins a specific uv version (`required-version = "==0.9.8"`). If your uv version doesn't match, downgrade first:

```bash
uv self update 0.9.8
```

### 2. Install Dependencies

The lockfile may be out of date after pulling new commits. If `--locked` fails, regenerate it first:

```bash
# Regenerate lockfile if needed
uv lock

# Install all dependencies including GPU extras
uv sync --extra gpu --group test
```

> Do NOT use `--locked` if the lockfile was just regenerated.

### 3. Install apex (for fused layer norm)

Some features (e.g. async Ulysses QKV projection) require NVIDIA apex's `fused_layer_norm_cuda` extension. It is not in `pyproject.toml` and must be built from source.

```bash
git clone https://github.com/NVIDIA/apex
cd apex
```

If the system CUDA version doesn't exactly match the PyTorch CUDA version (minor mismatch is OK), comment out the version check in `setup.py` around line 84:

```python
# Change raise RuntimeError(...) to a print warning
if bare_metal_version != torch_binary_version:
    print("WARNING: minor CUDA version mismatch, proceeding anyway.")
```

Then build:

```bash
/data/home/johnson/VeOmni/.venv/bin/python setup.py install --cpp_ext --cuda_ext
```

> Note: `uv pip` and `python -m pip` are not available in this venv. Use the venv Python directly.

---

## Running Tests

### Parallel / Ulysses Tests

Must be run from the **project root** via pytest (relative imports require package context):

```bash
cd ~/VeOmni
python -m pytest tests/parallel/ulysses/test_async_ulysses.py -v
```

> Running `python tests/parallel/ulysses/test_async_ulysses.py` directly will fail with `ImportError: attempted relative import with no known parent package`.

### E2E Parallel Alignment Tests

```bash
cd ~/VeOmni
python -m pytest tests/e2e/test_e2e_parallel.py -v
```

Run a specific model:

```bash
python -m pytest tests/e2e/test_e2e_parallel.py -v -k "llama3"
```

#### GPU Requirements

The e2e tests use `torchrun` with `--nproc_per_node = sp_size * 4`:

| sp_size | GPUs needed |
|---------|-------------|
| 1       | 4           |
| 2       | 8           |

- Non-MoE models: runs sp=1 and sp=2 → needs **8 GPUs**
- MoE models: runs sp×ep combinations → needs **8 GPUs**

If you only have 4 GPUs, edit `tests/e2e/utils.py` line 134 to limit sp_size:

```python
_SP_SIZE = [1]  # was [1, 2]
```

#### transformers Version

- Most tests require transformers v4 (`_v4_only`) — default install is `4.57.3` ✅
- `qwen3_5` requires transformers v5 (`_v5_only`) — skipped by default

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Required uv version ==0.9.8 does not match` | uv version mismatch | `uv self update 0.9.8` |
| `lockfile needs to be updated` | lockfile out of date | `uv lock` then `uv sync --extra gpu` |
| `No module named 'torch'` | sync failed, packages not installed | Re-run `uv sync --extra gpu --group test` |
| `No module named 'pytest'` | test group not installed | `uv sync --extra gpu --group test` |
| `attempted relative import with no known parent package` | running test file directly | Use `python -m pytest` from project root |
| `No module named 'fused_layer_norm_cuda'` | apex not installed | Build apex from source (see above) |
| `Cuda version does not match Pytorch binaries` | CUDA toolkit minor version mismatch | Comment out version check in `apex/setup.py:84` |
