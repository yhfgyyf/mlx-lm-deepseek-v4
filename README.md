# mlx-lm served-model-name patch

This repository vendors the Python package module `mlx_lm` from `mlx-lm==0.31.3` with a local patch that adds `--served-model-name` support to the package server.

## Contents

- `mlx_lm/`: patched package source snapshot
- `patches/served-model-name.patch`: diff from the upstream PyPI wheel source to this patched snapshot
- `metadata.json`: package/version/source metadata

## Apply Patch Later

From a checkout or unpacked wheel containing the upstream `mlx_lm` package directory:

```bash
git apply patches/served-model-name.patch
```

If upstream moved code around, inspect the rejected hunks and apply the same logic manually.
