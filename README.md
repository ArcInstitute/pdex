# pdex

parallel differential expression for single-cell perturbation sequencing

## Installation

Add to your `pyproject.toml` file with [`uv`](https://github.com/astral-sh/uv)

```bash
uv add pdex
```

## Summary

This is a python package for performing parallel differential expression between multiple groups and a control.

It is optimized for very large datasets and very large numbers of perturbations.

It makes use of shared memory to parallelize the computation to a high number of threads and minimizes the [IPC](https://en.wikipedia.org/wiki/Inter-process_communication) between processes to reduce overhead.

It supports the following metrics:

- Wilcoxon Rank Sum
- Anderson-Darling
- T-Test

## Backed vs In-Memory AnnData

pdex adapts its execution strategy based on how the AnnData object is stored:

- **In-memory AnnData** (e.g., loaded via `sc.read_h5ad(path)` without `backed="r"`):
  pdex uses a shared-memory multiprocessing workflow. Each worker process gets
  access to the full expression matrix through shared memory, which minimizes
  serialization overhead. Parallelism is configured via the `num_workers`
  parameter (process count). `num_threads` is ignored in this mode because numba
  kernels operate on per-target slices entirely in memory.
- **Backed AnnData** (`adata.X` is an on-disk HDF5 dataset): pdex automatically
  switches to the low-memory chunked implementation. Gene chunks are streamed
  from disk, the reference group is computed once per chunk, and targets are
  processed in parallel via `num_workers` (thread pool). Within each target,
  Wilcoxon metrics can additionally use numba parallelization controlled by
  `num_threads`. This mode avoids loading the entire matrix into RAM while still
  enabling both target-level and gene-level parallelism.

If a backed dataset is supplied without enabling low-memory mode, pdex raises a
helpful error explaining that chunked processing is required. Conversely, you can
force the chunked path for large in-memory matrices by passing `low_memory=True`.

## Parallelization

`parallel_differential_expression` exposes two orthogonal knobs for controlling
parallel execution:

- `num_workers` controls the number of Python threads that process targets within
  each gene chunk. `None` (default in low-memory mode) enables an auto-detected
  worker count based on available CPUs, while `1` disables thread-level parallelism.
- `num_threads` controls the numba thread pool used by the Wilcoxon kernel. `None`
  lets numba auto-detect the optimal size, whereas `1` turns numba parallelization
  off. This setting is only used in low-memory mode and only when `metric="wilcoxon"`.
  When pdex detects non-integer expression values in a gene chunk (for example, after
  log-normalization), it automatically disables numba for that chunk, logs a warning,
  and falls back to the SciPy implementation to preserve correct rank ordering.

These strategies can be combined: for example, `num_workers=2, num_threads=8`
runs two target threads that share an eight-thread numba pool. When the metric
does not support numba acceleration, pdex automatically logs a warning and
falls back to thread-only execution.

## Usage

```python
import anndata as ad
import numpy as np
import pandas as pd

from pdex import parallel_differential_expression

PERT_COL = "perturbation"
CONTROL_VAR = "control"

N_CELLS = 1000
N_GENES = 100
N_PERTS = 10
MAX_UMI = 1e6


def build_random_anndata(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_perts: int = N_PERTS,
    pert_col: str = PERT_COL,
    control_var: str = CONTROL_VAR,
) -> ad.AnnData:
    """Sample a random AnnData object."""
    return ad.AnnData(
        X=np.random.randint(0, MAX_UMI, size=(n_cells, n_genes)),
        obs=pd.DataFrame(
            {
                pert_col: np.random.choice(
                    [f"pert_{i}" for i in range(n_perts)] + [control_var],
                    size=n_cells,
                    replace=True,
                ),
            }
        ),
    )


def main():
    adata = build_random_anndata()

    # Run pdex with default metric (wilcoxon)
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
    )
    assert results.shape[0] == N_GENES * N_PERTS

    # Run pdex with alt metric (anderson)
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="anderson"
    )
    assert results.shape[0] == N_GENES * N_PERTS
```
