# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pdex2` is a Python library for Parallel Differential Expression (PDEX) analysis in single-cell genomics, focused on conditional screens.
It computes per-gene statistics comparing perturbation groups against a reference using Mann-Whitney U tests with FDR correction.
It also provides fundtionality for per-gene statistics on 1vRest comparisons.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_pdex.py

# Run a single test by name
uv run pytest tests/test_pdex.py::TestPdexRefMode::test_columns

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check
```

## Architecture

### Core Pipeline (`src/pdex2/__init__.py`)

The main entry point is `pdex(adata, groupby, mode, n_threads)`, which:

1. Validates the `groupby` column in `adata.obs`
2. Extracts unique groups (filters NaN and empty strings)
3. Identifies a reference group (defaults to `"non-targeting"` in `"ref"` mode)
4. For each group, slices the expression matrix, computes pseudobulk (mean), fold change, percent change, and Mann-Whitney U statistic vs the reference
5. Applies FDR correction (scipy) and returns a Polars DataFrame

Two modes:

- `"ref"`: each group vs a single reference group (e.g. non-targeting controls)
- `"all"`: each group vs all remaining cells (1-vs-rest)

### Key Files

| File                    | Role                                                                         |
| ----------------------- | ---------------------------------------------------------------------------- |
| `src/pdex2/__init__.py` | `pdex()` entry point and full pipeline logic                                 |
| `src/pdex2/_math.py`    | Numba JIT-compiled `fold_change()`, `percent_change()`, and `mwu()` wrappers |
| `src/pdex2/_utils.py`   | `set_numba_threadpool()` — sets Numba thread count before JIT warmup         |

### Performance Design

- Numba JIT compilation accelerates per-cell/per-gene math (`fold_change`, `percent_change`)
- `numba-mwu` (external Git dep) provides a Numba-accelerated Mann-Whitney U implementation
- Sparse CSR matrices are handled by reusing pre-computed non-targeting column indices to avoid redundant dense conversion
- Parallelism is controlled via `n_threads` passed to `set_numba_threadpool()`

### Output Schema

The returned Polars DataFrame has columns: `group`, `group_mean`, `ref_mean`, `group_membership`, `ref_membership`, `fold_change`, `percent_change`, `p_value`, `statistic`, `fdr`.

## Dependencies

Managed with `uv`. Key packages: `anndata`, `numpy`, `polars`, `scipy`, `tqdm`, `numba-mwu` (from GitHub via SSH). Dev tools: `pytest`, `ruff`, `ty`.
