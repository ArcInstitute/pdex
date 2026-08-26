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

The main entry point is `pdex(adata, groupby, mode, threads, is_log1p, geometric_mean, as_pandas, epsilon, cpm_filter, block_size, **kwargs)`, which:

1. Validates the `groupby` column in `adata.obs`
2. Extracts unique groups (filters NaN and empty strings)
3. Identifies a reference group (defaults to `"non-targeting"` in `"ref"` and `"on_target"` modes)
4. For each non-reference group, slices the expression matrix, computes pseudobulk (mean), fold change, percent change, and Mann-Whitney U statistic vs the reference
5. Optionally drops genes below the `cpm_filter` floor (see below), then applies FDR correction over the surviving genes (scipy) and returns a Polars DataFrame (or pandas if `as_pandas=True`)

Three modes:

- `"ref"`: each non-reference group vs a single reference group (reference group is excluded from output)
- `"all"`: each group vs all remaining cells (1-vs-rest)
- `"on_target"`: each non-reference group vs the reference, but only at the single gene targeted by that group (requires `gene_col=` kwarg)

Unexpected `**kwargs` for any mode trigger a `UserWarning`.

### Lazy / backed AnnData support

`pdex()` accepts three AnnData flavors transparently, with identical results:

- **in-memory** — ndarray or CSR sparse `X`
- **backed h5ad** — `ad.read_h5ad(path, backed="r")` (h5py `Dataset` / anndata `_CSRDataset`)
- **lazy** — `anndata.experimental.read_lazy(path_or_store)` over **h5ad or zarr, local or
  remote** (fsspec URLs like `s3://…` — the store genericity lives in anndata/zarr/fsspec,
  not in pdex). Lazy `X` is a dask array (dense or scipy-sparse chunks); obs/var are
  xarray-backed `Dataset2D`.

All `X` access funnels through `src/pdex/_backend.py`:

- `realize(x)` — materializes any backend slice to `ndarray`/`csr_matrix` (dask
  `.compute()`, sparse → CSR, h5py/zarr → `np.asarray`). Duck-typed; dask is never
  imported (lazy IO deps are the optional `pdex[lazy]` extra: `anndata[lazy]`, zarr, fsspec).
- `is_lazy(x)` — dask detection. Dask disallows mixed fancy-row + scalar-column indexing,
  so `_isolate_matrix` indexes lazy `X` in two steps and keeps the column axis 2-D.
- `default_block_size(x, n_other, axis)` — chunk-aligned block extent along `axis`
  targeting ~256 MB dense-equivalent per block; returns `None` for in-memory `X`.

**Never fancy-row-index a lazy (dask) X per group** — scattered groups touch nearly every
storage chunk, so dask re-reads the dataset ~once *per group* (measured 10x slowdown at
100k cells x 100 groups; scales with group count). All lazy row access instead goes through
`_stream_row_groups(x, n_obs, row_groups)`: one pass over chunk-aligned obs blocks, each
block realized exactly once, rows scattered to the requesting groups (blocks no group
needs are never computed). Per-mode strategy on lazy X:

- `"ref"` — reference realized via one streaming pass; target groups batched into
  memory-budgeted **rounds** (`_stream_budget_bytes()`: 25% of available RAM clamped to
  [512 MiB, 64 GiB], falling back to `_STREAM_BUDGET_BYTES` = 4 GiB when unknown; group
  bytes estimated from the realized reference's bytes/row), one streaming pass per round.
  Total passes ≈ `total_group_bytes / budget` instead of one per group; peak memory ≈
  budget. The tqdm bar shows `streaming round i/n` while multi-round.
- `"all"` — single-block: one `_stream_row_groups` pass; var-blocked: `realize(X[:, j0:j1])`
  per block (cheap on var-chunked stores) with the valid-row filter applied in memory.
- `"on_target"` — one streaming pass over `X[:, needed_genes]` collects every group's
  target-gene column and the reference rows at all needed genes (`fetch_cols` closure).

Backed h5ad (h5py/`_CSRDataset`) keeps the original per-group row slicing — direct reads,
no dask amplification. `_per_cell_library_sizes` (on_target + cpm_filter) streams
contiguous **row blocks**. obs columns from `Dataset2D` are normalized via `_obs_series()`
before pandas ops. `_detect_is_log1p` and `_x_has_negative` realize their bounded samples.
Tests: `tests/test_lazy.py` asserts lazy/backed == in-memory across all modes, block-size
invariance, forced multi-block/multi-round streaming (monkeypatched `default_block_size` /
`_STREAM_BUDGET_BYTES`), and an fsspec `memory://` store (same code path as s3).

### CPM floor filter (`cpm_filter`)

`cpm_filter` (default `None` = off) is an opt-in **two-view** floor filter. A native
(unnormalized) view drives the reported means, LFC, MWU, and output; a separate **CPM
view** drives only the keep/drop decision, so the output is never normalized. Per group
per gene the pooled (bulk) CPM is `Σcounts_gene / Σcounts_all * 1e6` (computed on counts —
`expm1` is applied first when `is_log1p`). A `(target, gene)` row is **dropped** when the
gene's CPM is `<= T` in **both** the target and the reference (kept iff `target_cpm > T`
**or** `ref_cpm > T`; strict `>`, negative `T` keeps everything). The CPM ratio is
scale-invariant, so `T` means the same regardless of input normalization. The drop is
independent of the MWU result, and **FDR is corrected over the surviving genes only**.
Applies to all three modes (in `on_target`, a group whose target gene is a floor gene is
dropped). Emits a `UserWarning` if the data contains negative values. `T = 5` is a
reasonable starting point, but the optimal threshold is dataset-dependent and should be
checked empirically (inspect the per-gene CPM distribution).

### Key Files

| File                   | Role                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `src/pdex/__init__.py` | `pdex()` entry point and full pipeline logic                                                            |
| `src/pdex/_math.py`    | Numba JIT-compiled `log2_fold_change()`, `percent_change()`, and `mwu()`/`mwu_one_vs_rest()` wrappers over `numba-mwu`; `pseudobulk()` dispatcher; `cpm_bulk()` pooled-CPM view for the filter; the `bulk_matrix_pre_transform_mean()`/`pseudobulk_from_pre_mean()`/`cpm_from_gene_means()` trio powering `"all"` mode's global-sum optimization |
| `src/pdex/_utils.py`   | `set_numba_threadpool()` — sets Numba thread count before JIT warmup; `_available_cpus()` — affinity-aware CPU count (respects cgroup/SLURM limits); `_detect_is_log1p()` heuristic |
| `src/pdex/_backend.py` | Storage-backend normalization: `realize()` (any slice → ndarray/csr), `is_lazy()` (dask detection), `default_block_size()` (chunk-aligned streaming block extents) |

### Performance Design

- Numba JIT compilation accelerates per-cell/per-gene math (`log2_fold_change`, `percent_change`, `_log1p_col_mean`, `_expm1_vec`)
- `numba-mwu` (external dep, `>=0.2.0`) provides Numba-accelerated Mann-Whitney U kernels for **both** the pairwise case (`mannwhitneyu_columns`/`mannwhitneyu_sparse`, used by `"ref"` and `"on_target"` modes) and the one-vs-rest case (`mannwhitneyu_one_vs_rest`/`_sparse`, used by `"all"` mode) — the one-vs-rest kernels originated in pdex and were upstreamed into `numba-mwu` since the optimization is domain-agnostic (see that package's CLAUDE.md for the algorithm).
- Sparse CSR matrices are handled by reusing pre-computed non-targeting column indices to avoid redundant dense conversion
- Parallelism is controlled via `threads` passed to `set_numba_threadpool()`
- **`"all"` mode (1-vs-rest) is a one-shot computation, not a per-group loop over `numba-mwu`.** Because `group ∪ rest` is always the full (non-filtered) dataset regardless of which group is being tested, `_pdex_all` reads each value exactly once (instead of once per group) and:
  - Ranks each gene once across all cells via `mwu_one_vs_rest()` (`_math.py`, a thin wrapper over `numba_mwu.mannwhitneyu_one_vs_rest`/`_sparse`) and reduces to every group's rank-sum in the same pass, rather than re-ranking group+rest from scratch per group.
  - Derives each group's "rest" pseudobulk and CPM algebraically as `(global_sum - group_sum) / n_rest` (`bulk_matrix_pre_transform_mean()`/`pseudobulk_from_pre_mean()`/`cpm_from_gene_means()`) instead of recomputing over a freshly sliced "rest" matrix.
  - This turns `_pdex_all` from `O(n_groups × n_obs)` matrix I/O and ranking into ~`O(n_obs)` total, which matters most for screens with many groups (e.g. guides) and/or large cell counts.
  - The read happens in **var (gene) blocks** (`block_size` param): every statistic (pre-transform mean, arithmetic mean, one-vs-rest ranking) is column-independent, so per-block results concatenate to bit-identical full-matrix results. `block_size=None` (default) resolves to a single block for in-memory `X` (previous behaviour) and chunk-aligned ~256 MB blocks for lazy `X`, so the full matrix is never resident. **Trap:** CPM normalizes across all genes — it is computed only after all blocks' arithmetic means are reduced, never per block.

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
| `log2_fold_change`  | float | log2((target_mean + epsilon) / (ref_mean + epsilon)) — computed from pseudobulk means. `epsilon` defaults to `1e-9` (finite-guard), so by default there are no `±inf`/`NaN`: one-sided zeros become large-but-finite and `0/0` is `0.0`. With `epsilon == 0`, `0/0` is still defined as `0.0` (not `NaN`) but one-sided zeros yield `±inf`. |
| `percent_change`    | float | (target_mean - ref_mean) / (ref_mean + epsilon) — computed from pseudobulk means. With the default `epsilon=1e-9` there are no non-finite values; with `epsilon == 0`, `0/0` is `0.0` (not `NaN`) and a zero reference with nonzero target yields `+inf`. |
| `p_value`           | float | Mann-Whitney U p-value (per-cell vectors)                             |
| `statistic`         | float | Mann-Whitney U statistic                                              |
| `fdr`               | float | FDR-corrected p-value, applied per-group across genes. For `on_target` mode, applied across all groups. When `cpm_filter` is set, the correction is over the **surviving** genes only.                 |

`target_mean` and `ref_mean` are always in natural (count) space regardless of `is_log1p` or `geometric_mean`.
FDR is corrected within each group (across genes) for `ref` and `all` modes. For `on_target` mode, it is applied across all resulting p-values.
`epsilon` defaults to `1e-9`; pass `epsilon=0.0` to recover legacy `±inf` for one-sided zeros.
When `cpm_filter` is set, genes failing the floor are **dropped**, so the output has fewer than `n_targets × n_genes` rows (a fully-filtered comparison yields a height-0 frame with the full schema).

### Public API (`__all__`)

```python
from pdex import pdex, DEFAULT_REFERENCE
```

## Dependencies

Managed with `uv`. Build backend: `hatchling`. Key packages: `anndata`, `numba`, `numba-mwu`, `polars`, `pyarrow`, `scipy`, `tqdm`. Optional extra `pdex[lazy]` (lazy/remote IO, never imported at module scope): `anndata[lazy]` (dask, xarray), `zarr`, `fsspec`. Dev tools: `pytest`, `ruff`, `ty` (dev group includes `pdex[lazy]` so the lazy tests run).
