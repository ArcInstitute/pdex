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
