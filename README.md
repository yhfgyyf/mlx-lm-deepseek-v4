# mlx-lm-deepseek-v4

This repository is an experimental fork of
[ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) focused on running
large DeepSeek V4 / MoE MLX models on Apple silicon with layer-level expert
disk offload.

The current branch includes upstream DeepSeek V4 support plus local changes for
OpenAI-compatible serving, model-name aliases, and MoE expert offload sizing.

## Installation

This fork is packaged as `mlx-lm-deepseek-v4`. It installs the same Python
module and CLI entry points as upstream MLX LM: `mlx_lm`, `mlx_lm.server`,
`mlx_lm.generate`, and the rest of the `mlx_lm.*` commands.

### Install From GitHub Releases

Create a Python environment and install the release wheel:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  https://github.com/yhfgyyf/mlx-lm-deepseek-v4/releases/download/v0.31.3-deepseek-v4/mlx_lm_deepseek_v4-0.31.3-py3-none-any.whl
```

After installation, verify the CLI is available:

```bash
mlx_lm.server --help
python -c "import mlx_lm; print(mlx_lm.__version__)"
```

### Install From the Source Checkout

For development, clone the repository and install it editable:

```bash
git clone https://github.com/yhfgyyf/mlx-lm-deepseek-v4.git
cd mlx-lm-deepseek-v4
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## What Changed

- Applied upstream DeepSeek V4 model support.
- Added `--served-model-name` to expose a stable public model id from
  `/v1/models` and chat/completion responses while loading weights from a local
  path.
- Added MoE routed-expert disk offload for stacked MLX safetensors layouts.
- Added `--n-disk-moe N` to offload the last `N` MoE layers.
- Added `--n-disk-moe auto` to choose how many prefix MoE layers stay resident
  based on load-time memory.
- Added macOS-aware memory sizing that counts only part of inactive memory by
  default, avoiding the Metal OOMs seen when inactive memory is treated as fully
  usable.
- Added DeepSeek V4 raw expert weight remapping so mixed resident/offloaded
  expert layers can load correctly.

## Auto MoE Sizing

`--n-disk-moe auto` computes:

```text
available memory
- non-MoE resident weights
- system reserve
- runtime reserve
= resident MoE budget
```

On macOS, `available memory` is estimated as:

```text
free + speculative + purgeable + inactive * inactive_ratio
```

The default `inactive_ratio` is `0.5`. This deliberately uses only part of
inactive memory because MLX/Metal can fail with command-buffer OOM even when
macOS still reports reclaimable inactive pages.

Relevant server flags:

```bash
--n-disk-moe auto
--moe-expert-reserve-mb 4096
--moe-expert-runtime-reserve-mb 2048
--moe-expert-inactive-memory-ratio 0.5
--moe-expert-cache-mb 4096
```

If the model is too conservative, raise `--moe-expert-inactive-memory-ratio`.
If generation crashes with Metal OOM, lower it.

## Example Server

```bash
export MODEL_DIR=/path/to/deepseek-ai-DeepSeek-V4-Flash-3bit

mlx_lm.server \
  --host 127.0.0.1 \
  --port 8081 \
  --model "$MODEL_DIR" \
  --served-model-name deepseek-v4-flash-3bit \
  --max-tokens 256 \
  --n-disk-moe auto \
  --moe-expert-reserve-mb 4096 \
  --moe-expert-runtime-reserve-mb 2048 \
  --moe-expert-inactive-memory-ratio 0.5 \
  --moe-expert-cache-mb 4096 \
  --trust-remote-code
```

The server logs the auto decision at startup, for example:

```text
--n-disk-moe auto: available=17.32 GiB, non_moe=3.18 GiB,
system_reserve=4.00 GiB, runtime_reserve=2.00 GiB,
resident_moe_budget=8.15 GiB, resident_moe=7.88 GiB,
resident_cost_scale=1.00, inactive_memory_ratio=0.50,
resident_layers=3, offload_layers=40
```

## Smoke Test

```bash
curl http://127.0.0.1:8081/v1/models

curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-3bit",
    "messages": [{"role": "user", "content": "只输出 OK"}],
    "max_tokens": 4,
    "temperature": 0
  }'
```

## Recent Local Benchmark

Environment: local DeepSeek V4 Flash 3-bit MLX model, server with
`--n-disk-moe auto`, `--moe-expert-inactive-memory-ratio 0.5`.

Prompt/output:

- Input: about 1024 tokens under the DeepSeek tokenizer/chat template
- Output: 32 tokens

Result:

```text
TTFT: 30.10 s
Decode time: 17.63 s
Total time: 47.72 s
Decode TPS: 1.82 tok/s
Overall TPS: 0.67 tok/s
```

## Validation

The current branch was checked with:

```bash
python -m unittest -v tests.test_moe_disk_offload tests.test_server.TestServedModelName
python -m py_compile mlx_lm/models/moe_disk_offload.py mlx_lm/utils.py mlx_lm/server.py
```

## Notes

This is a Python-layer prototype. It does not implement llama.cpp-style
zero-copy mmap execution in MLX core. Offloaded expert slices are read from
safetensors on demand and kept in an LRU cache.

For best stability, tune `--moe-expert-inactive-memory-ratio` together with the
reserve flags on the target machine and prompt length.
