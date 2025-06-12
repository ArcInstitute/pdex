import anndata as ad
import numpy as np
import pandas as pd
import polars as pl

from pdex import pseudobulk_dex

PERT_COL = "perturbation"
CONTROL_VAR = "control"

N_CELLS = 1000
N_GENES = 100
N_PERTS = 10
MAX_UMI = 1e6
N_CONDITIONS = 4

RANDOM_SEED = 42


def build_random_anndata(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_perts: int = N_PERTS,
    pert_col: str = PERT_COL,
    control_var: str = CONTROL_VAR,
    n_covariates: int = 0,
    covariate_magnitudes: list[int] = [],
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Sample a random AnnData object."""
    if random_state is not None:
        np.random.seed(random_state)
    obs = pd.DataFrame(
        {
            pert_col: np.random.choice(
                [f"pert_{i}" for i in range(n_perts)] + [control_var],
                size=n_cells,
                replace=True,
            ),
        },
        index=np.arange(n_cells).astype(str),
    )

    for idx in np.arange(n_covariates):
        n_conditions = (
            N_CONDITIONS
            if len(covariate_magnitudes) == 0
            else covariate_magnitudes[idx]
        )
        obs[f"covar.{idx}"] = [
            f"enum.{i}" for i in np.random.choice(n_conditions, size=n_cells)
        ]

    return ad.AnnData(
        X=np.random.randint(0, int(MAX_UMI), size=(n_cells, n_genes)),
        obs=obs,
        var=pd.DataFrame(index=np.array([f"gene.{j}" for j in np.arange(N_GENES)])),
    )


def test_pbdex_no_covar():
    adata = build_random_anndata()
    try:
        _ = pseudobulk_dex(
            adata, groupby=[PERT_COL], reference=CONTROL_VAR, test_col=PERT_COL
        )
        assert False, "Should not arrive here"
    except ValueError:
        assert True


def test_pbdex_empty_groupby():
    adata = build_random_anndata()
    try:
        _ = pseudobulk_dex(adata, groupby=[], reference=CONTROL_VAR, test_col=PERT_COL)
        assert False, "Should not arrive here"
    except ValueError:
        assert True


def test_pbdex_one_covar():
    adata = build_random_anndata(n_covariates=1)
    results = pseudobulk_dex(
        adata, reference=CONTROL_VAR, test_col=PERT_COL, groupby=[PERT_COL, "covar.0"]
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_pbdex_two_covar():
    adata = build_random_anndata(n_covariates=2)
    results = pseudobulk_dex(
        adata,
        reference=CONTROL_VAR,
        test_col=PERT_COL,
        groupby=[PERT_COL, "covar.0", "covar.1"],
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_pbdex_polars_output():
    adata = build_random_anndata(n_covariates=1)
    results = pseudobulk_dex(
        adata,
        reference=CONTROL_VAR,
        test_col=PERT_COL,
        groupby=[PERT_COL, "covar.0"],
        as_polars=True,
    )
    assert results.shape[0] == N_GENES * N_PERTS
    assert isinstance(results, pl.DataFrame)
