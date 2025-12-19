import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
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
        X=np.random.randint(0, int(MAX_UMI), size=(n_cells, n_genes)),
        obs=pd.DataFrame(
            {
                pert_col: np.random.choice(
                    [f"pert_{i}" for i in range(n_perts)] + [control_var],
                    size=n_cells,
                    replace=True,
                ),
            },
            index=np.arange(n_cells).astype(str),
        ),
    )


def build_small_anndata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    n_cells_per_group = 48
    n_genes = 5
    control = rng.integers(0, 50, size=(n_cells_per_group, n_genes))
    pert = rng.integers(0, 50, size=(n_cells_per_group, n_genes))
    X = np.vstack([control, pert]).astype(np.int32)
    obs = pd.DataFrame(
        {PERT_COL: [CONTROL_VAR] * n_cells_per_group + ["pert_a"] * n_cells_per_group},
        index=pd.Index([f"cell_{i}" for i in range(X.shape[0])]),
    )
    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(n_genes)]))
    return ad.AnnData(X=X, obs=obs, var=var, dtype=X.dtype)


def test_dex_dense_array():
    adata = build_random_anndata()
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_dense_array_log():
    adata = build_random_anndata()
    adata.X = np.log1p(adata.X)  # type: ignore
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_dense_array_log_post_agg():
    adata = build_random_anndata()
    adata.X = np.log1p(adata.X)  # type: ignore
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        exp_post_agg=True,
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_dense_matrix():
    adata = build_random_anndata()
    adata.X = np.matrix(adata.X)  # type: ignore
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


def test_dex_wilcoxon():
    adata = build_random_anndata()
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_wilcoxon_no_tie_correct():
    adata = build_random_anndata()
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
        tie_correct=False,
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_anderson():
    adata = build_random_anndata()
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="anderson",
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_ttest():
    adata = build_random_anndata()
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="t-test",
    )
    assert results.shape[0] == N_GENES * N_PERTS


def test_dex_unknown_metric():
    adata = build_random_anndata()
    try:
        parallel_differential_expression(
            adata,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            metric="unknown",
        )
        assert False
    except ValueError:
        "Caught error"


def test_dex_single_observed_value_anderson():
    adata = build_random_anndata()
    adata.X = np.zeros_like(adata.X)
    parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="anderson",
    )


def test_dex_polars_output():
    adata = build_random_anndata()
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        as_polars=True,
    )
    assert results.shape[0] == N_GENES * N_PERTS
    assert isinstance(results, pl.DataFrame)


def test_zeroed_matrix():
    adata = build_random_anndata()
    adata.X = np.zeros_like(adata.X)
    results = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
    )
    assert results.shape[0] == N_GENES * N_PERTS
    assert isinstance(results, pd.DataFrame)
    assert np.all(results["p_value"] == 1.0)
    assert np.all(results["fdr"] == 1.0)


def _sort_results(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    return df.sort_values(["target", "feature"]).reset_index(drop=True)


def test_low_memory_integer_data_matches_standard_results():
    adata = build_small_anndata()

    baseline = _sort_results(
        parallel_differential_expression(
            adata,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            num_workers=1,
            low_memory=False,
        )
    )

    chunked = _sort_results(
        parallel_differential_expression(
            adata,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            num_workers=1,
            num_threads=2,
            low_memory=True,
            gene_chunk_size=2,
            show_progress=False,
        )
    )

    pd.testing.assert_series_equal(
        baseline["p_value"],
        chunked["p_value"],
        check_exact=False,
        rtol=0,
        atol=5e-2,
    )
    pd.testing.assert_series_equal(baseline["statistic"], chunked["statistic"])


def test_low_memory_float_data_uses_sorting_kernel():
    adata = build_small_anndata()
    adata_float = adata.copy()
    adata_float.X = np.log1p(np.asarray(adata_float.X))  # type: ignore[assignment]

    baseline = _sort_results(
        parallel_differential_expression(
            adata_float,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            num_workers=1,
            low_memory=False,
        )
    )

    chunked = _sort_results(
        parallel_differential_expression(
            adata_float,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            num_workers=1,
            num_threads=2,
            low_memory=True,
            gene_chunk_size=2,
            show_progress=False,
        )
    )

    # Allow small numerical differences between scipy and numba implementations
    pd.testing.assert_series_equal(
        baseline["p_value"],
        chunked["p_value"],
        check_exact=False,
        rtol=0,
        atol=5e-3,
    )
    pd.testing.assert_series_equal(
        baseline["statistic"],
        chunked["statistic"],
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
