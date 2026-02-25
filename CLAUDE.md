# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Important:** This file must be kept up to date with the codebase. Any time the public API, output schema, modes, parameters, or architecture changes, update the relevant sections here before closing the task.

## Project Overview

`pdex` is a Python library for Parallel Differential Expression (PDEX) analysis in single-cell genomics, focused on conditional screens.
It computes per-gene statistics comparing perturbation groups against a reference using Mann-Whitney U tests with FDR correction.
It also provides functionality for per-gene statistics on 1-vs-rest comparisons and on-target single-gene comparisons.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_pdex.py

# Run a single test by name
uv run pytest tests/test_pdex.py::TestPdexRefMode::test_columns

# Lint and format
uv run ruff format

# Type check
uv run ty check
```

## Architecture

### Core Pipeline (`src/pdex/__init__.py`)

The main entry point is `pdex(adata, groupby, mode, threads, is_log1p, geometric_mean, as_pandas, **kwargs)`, which:

1. Validates the `groupby` column in `adata.obs`
2. Extracts unique groups (filters NaN and empty strings)
3. Identifies a reference group (defaults to `"non-targeting"` in `"ref"` and `"on_target"` modes)
4. For each non-reference group, slices the expression matrix, computes pseudobulk (mean), fold change, percent change, and Mann-Whitney U statistic vs the reference
5. Applies per-group FDR correction (scipy) and returns a Polars DataFrame (or pandas if `as_pandas=True`)

Three modes:

- `"ref"`: each non-reference group vs a single reference group (reference group is excluded from output)
- `"all"`: each group vs all remaining cells (1-vs-rest)
- `"on_target"`: each non-reference group vs the reference, but only at the single gene targeted by that group (requires `gene_col=` kwarg)

Unexpected `**kwargs` for any mode trigger a `UserWarning`.

### Key Files

| File                   | Role                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `src/pdex/__init__.py` | `pdex()` entry point and full pipeline logic                                                            |
| `src/pdex/_math.py`    | Numba JIT-compiled `fold_change()`, `percent_change()`, and `mwu()` wrappers; `pseudobulk()` dispatcher |
| `src/pdex/_utils.py`   | `set_numba_threadpool()` — sets Numba thread count before JIT warmup; `_detect_is_log1p()` heuristic    |

### Performance Design

- Numba JIT compilation accelerates per-cell/per-gene math (`fold_change`, `percent_change`, `_log1p_col_mean`, `_expm1_vec`)
- `numba-mwu` (external dep) provides a Numba-accelerated Mann-Whitney U implementation
- Sparse CSR matrices are handled by reusing pre-computed non-targeting column indices to avoid redundant dense conversion
- Parallelism is controlled via `threads` passed to `set_numba_threadpool()`

### Output Schema

The returned Polars DataFrame (or pandas DataFrame when `as_pandas=True`) has columns:

| Column              | Type  | Description                                                           |
| ------------------- | ----- | --------------------------------------------------------------------- |
| `target`            | str   | Perturbation group name                                               |
| `feature`           | str   | Gene name                                                             |
| `target_mean`       | float | Pseudobulk mean for the target group, always in natural (count) space |
| `ref_mean`          | float | Pseudobulk mean for the reference, always in natural (count) space    |
| `target_membership` | int   | Number of cells in the target group                                   |
| `ref_membership`    | int   | Number of cells in the reference                                      |
| `fold_change`       | float | log2(target_mean / ref_mean) — computed from pseudobulk means         |
| `percent_change`    | float | (target_mean - ref_mean) / ref_mean — computed from pseudobulk means  |
| `p_value`           | float | Mann-Whitney U p-value (per-cell vectors)                             |
| `statistic`         | float | Mann-Whitney U statistic                                              |
| `fdr`               | float | FDR-corrected p-value, applied per-group across genes                 |

`target_mean` and `ref_mean` are always in natural (count) space regardless of `is_log1p` or `geometric_mean`.
FDR is corrected within each group (across genes), not globally across all (group, gene) pairs.

### Public API (`__all__`)

```python
from pdex import pdex, DEFAULT_REFERENCE
```

## Dependencies

Managed with `uv`. Build backend: `hatchling`. Key packages: `anndata`, `numba`, `numba-mwu`, `polars`, `pyarrow`, `scipy`, `tqdm`. Dev tools: `pytest`, `ruff`, `ty`.
