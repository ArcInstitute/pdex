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


## Updates

This repo contains a few recent updates to test the performance of vectorized operations for statistical tests in comparison to multiprocessing batches.

The two function added are `_vectorized_ranksum_test` and `parallel_differential_expression_vec` in [src/pdex/_single_cell.py](src/pdex/_single_cell.py).

### Run benchmarks

```bash
uv run python -m pytest benches/basic.py
```

output:

```txt
=================================================================================================== test session starts ====================================================================================================
platform darwin -- Python 3.10.15, pytest-8.4.1, pluggy-1.6.0
benchmark: 5.1.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /Users/drbh/Projects/pdex
configfile: pyproject.toml
plugins: benchmark-5.1.0
collected 3 items

benches/basic.py ...                                                                                                                                                                                                 [100%]


--------------------------------------------------------------------------------------------------- benchmark: 2 tests --------------------------------------------------------------------------------------------------
Name (time in ms)                                  Min                   Max                  Mean             StdDev                Median                IQR            Outliers      OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_fast_implementation             51.4345 (1.0)         54.8240 (1.0)         52.3380 (1.0)       0.8007 (1.0)         52.1402 (1.0)       0.8673 (1.0)           4;1  19.1066 (1.0)          19           1
test_benchmark_reference_implementation     1,793.7180 (34.87)    1,840.0030 (33.56)    1,814.5615 (34.67)    18.8329 (23.52)    1,811.3458 (34.74)    30.7128 (35.41)         2;0   0.5511 (0.03)          5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
==================================================================================================== 3 passed in 17.60s ====================================================================================================
```


### Run comparison script

```bash
uv run scripts/compare.py
```

output

```txt  
============================================================
Benchmarking with 100 cells, 500 genes, 300 perturbations
============================================================

1. Reference implementation (batch processing):
INFO:pdex._single_cell:Precomputing masks for each target gene
Identifying target masks: 100%|██████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 29944.39it/s]
INFO:pdex._single_cell:Precomputing variable indices for each feature
Identifying variable indices: 100%|██████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 5637505.38it/s]
INFO:pdex._single_cell:Creating shared memory memory matrix for parallel computing
INFO:pdex._single_cell:Creating generator of all combinations: N=43000
INFO:pdex._single_cell:Creating generator of all batches: N=431
INFO:pdex._single_cell:Initializing parallel processing pool
INFO:pdex._single_cell:Processing batches
Processing batches: 100%|████████████████████████████████████████████████████████████████████████████▊| 430/431 [00:07<00:00, 55.84it/s]
INFO:pdex._single_cell:Flattening results
INFO:pdex._single_cell:Closing shared memory pool
   Time: 7.749 seconds

2. Vectorized implementation:
INFO:pdex._single_cell:vectorized processing: 86 targets, 500 genes
Processing targets: 100%|██████████████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 978.07it/s]
   Time: 0.117 seconds
   Speedup: 66.0x

============================================================
Correctness Verification:
============================================================
✅ vec: Column 'target_mean' values match within 1e-06 tolerance
✅ vec: Column 'reference_mean' values match within 1e-06 tolerance
✅ vec: Column 'percent_change' values match within 0.01 tolerance
✅ vec: Column 'fold_change' values match within 1e-06 tolerance
✅ vec: Results match reference

============================================================
Performance Summary:
============================================================
Implementation                 Time (s)     Speedup
----------------------------------------------------
reference                      7.749        1.0       x
vec                            0.117        66.0      x
```