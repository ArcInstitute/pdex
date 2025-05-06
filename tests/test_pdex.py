import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pdex import parallel_differential_expression

PERT_COL = "perturbation"
CONTROL_VAR = "control"

N_CELLS = 1000
N_GENES = 100
N_PERTS = 10
MAX_UMI = 1e6

RANDOM_SEED = 42


def build_random_anndata(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_perts: int = N_PERTS,
    pert_col: str = PERT_COL,
    control_var: str = CONTROL_VAR,
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Sample a random AnnData object."""
    if random_state is not None:
        np.random.seed(random_state)
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


def test_dex_dense_array():
    adata = build_random_anndata()
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_dense_matrix():
    adata = build_random_anndata()
    adata.X = np.matrix(adata.X)
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_sparse_matrix():
    adata = build_random_anndata()
    adata.X = csr_matrix(adata.X)
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
    )
    assert results.shape[0] == N_GENES * N_PERTS
