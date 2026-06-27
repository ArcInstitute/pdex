# pdex

Parallel differential expression for single-cell perturbation sequencing.

## Installation

```bash
# add to pyproject.toml
uv add pdex

# add to env
uv pip install pdex
```

## Overview

`pdex` computes per-gene differential expression statistics between perturbation groups in single-cell data using Mann-Whitney U tests with FDR correction. It was originally designed for CRISPR screen and perturbation sequencing datasets with many groups and large cell counts.

It supports dense and sparse (CSR) expression matrices, and uses [numba-mwu](https://github.com/noamteyssier/numba-mwu) for Numba-accelerated Mann-Whitney U computation.

## Modes

| Mode          | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| `"ref"`       | Each group vs a single reference group (default: `"non-targeting"`) |
| `"all"`       | Each group vs all remaining cells (1-vs-rest)                       |
| `"on_target"` | Each group vs the reference at its single target gene only          |

## Usage

### Reference mode (default)

```python
import anndata as ad
from pdex import pdex

adata = ad.read_h5ad("screen.h5ad")

results = pdex(
    adata,
    groupby="guide",
    mode="ref",
    is_log1p=False,
)
```

### 1-vs-rest mode

```python
results = pdex(
    adata,
    groupby="guide",
    mode="all",
    is_log1p=False,
)
```

### On-target mode

Requires a column in `adata.obs` mapping each group to its target gene:

```python
results = pdex(
    adata,
    groupby="guide",
    mode="on_target",
    gene_col="target_gene",
    is_log1p=False,
)
```

## Parameters

| Parameter        | Type           | Default           | Description                                                |
| ---------------- | -------------- | ----------------- | ---------------------------------------------------------- |
| `adata`          | `AnnData`      | required          | Annotated data matrix (dense or sparse CSR)                |
| `groupby`        | `str`          | required          | Column in `adata.obs` defining groups                      |
| `mode`           | `str`          | `"ref"`           | Comparison mode: `"ref"`, `"all"`, or `"on_target"`        |
| `threads`        | `int`          | `0`               | Numba thread count (`0` = all CPUs)                        |
| `is_log1p`       | `bool \| None` | `None`            | Whether data is log1p-transformed. Auto-detected if `None` |
| `geometric_mean` | `bool`         | `True`            | Use geometric mean for pseudobulk (vs arithmetic)          |
| `as_pandas`      | `bool`         | `False`           | Return a pandas DataFrame instead of Polars                |
| `epsilon`        | `float`        | `1e-9`            | Pseudocount used for `log2_fold_change` and `percent_change`; pass `0.0` for legacy one-sided `±inf` values |
| `cpm_filter`     | `float \| None` | `None`           | Optional pooled-CPM floor filter; drops rows where both target and reference CPM are at or below the threshold |
| `reference`      | `str`          | `"non-targeting"` | Reference group name (modes: `ref`, `on_target`)           |
| `gene_col`       | `str`          | —                 | Column mapping groups to target genes (mode: `on_target`)  |

### CPM filter

`cpm_filter` is an opt-in floor filter. When set to a threshold `T`, a `(target, feature)` row is dropped only when the gene's pooled CPM is `<= T` in both the target group and the reference group. Rows are kept when either side has CPM `> T`.

The CPM view is used only for filtering: reported means, fold changes, MWU statistics, and p-values are still computed from the original expression scale. When `is_log1p=True`, counts are recovered with `expm1` before CPM is computed. FDR is corrected over the surviving genes only.

## Output

Returns a Polars DataFrame (or pandas if `as_pandas=True`) with one row per (group, gene) pair:

| Column              | Description                                        |
| ------------------- | -------------------------------------------------- |
| `target`            | Perturbation group name                            |
| `feature`           | Gene name                                          |
| `target_mean`       | Pseudobulk mean for the target group (count space) |
| `ref_mean`          | Pseudobulk mean for the reference (count space)    |
| `target_membership` | Number of cells in the target group                |
| `ref_membership`    | Number of cells in the reference                   |
| `fold_change`       | **Deprecated alias** for `log2_fold_change` (identical values). Will be removed in pdex 0.3.0. |
| `log2_fold_change`  | log2((target_mean + epsilon) / (ref_mean + epsilon)). With default `epsilon=1e-9`, one-sided zeros are large finite values; with `epsilon=0.0`, one-sided zeros yield `±inf`. Genes unexpressed in both groups (0/0) report `0.0`, not `NaN`. |
| `percent_change`    | (target_mean - ref_mean) / (ref_mean + epsilon). With default `epsilon=1e-9`, zero-reference cases are finite; with `epsilon=0.0`, a zero reference with nonzero target yields `+inf`. Genes unexpressed in both groups (0/0) report `0.0`, not `NaN`. |
| `p_value`           | Mann-Whitney U p-value                             |
| `statistic`         | Mann-Whitney U statistic                           |
| `fdr`               | FDR-corrected p-value (per-group, across genes). For `on_target` mode, this is applied across all groups. When `cpm_filter` is set, FDR is corrected over surviving genes only. |
