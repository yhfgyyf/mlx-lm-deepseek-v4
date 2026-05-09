# mlx-lm MoE disk offload fork

This repository is an independent, git-managed fork snapshot of `mlx-lm`
derived from the `mlx-lm==0.31.3` Python package source.

The goal is to keep local serving changes as normal branches, commits, and pull
requests instead of distributing standalone patch files.

## Features

- OpenAI-compatible server `--served-model-name` support.
- Python-level MoE expert disk LRU offload for stacked Qwen MoE weights.
- `--n-disk-moe N`, which offloads the last `N` MoE layers to disk-backed LRU
  execution.
- Compatibility flag `--moe-expert-offload-layers` for explicit layer lists.

## Example

```bash
mlx_lm.server \
  --model /path/to/mlx/model \
  --served-model-name qwen-local \
  --n-disk-moe 20 \
  --moe-expert-cache-mb 4096
```

When `--n-disk-moe` is greater than zero, `disk-lru` offload is enabled
automatically and the selected layer set is the final `N` transformer layers.

## Development

All changes should be made through git branches and pull requests.

```bash
git switch -c codex/my-change
python -m unittest discover -s tests -v
git commit
git push -u origin codex/my-change
```
