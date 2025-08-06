# Run with: `uv run python -m pytest tests/bench_expr.py`
import sys
import importlib
import pytest
import numpy as np
import pandas as pd
import anndata as ad


def _reload_pdex():
    """Reload pdex modules to pick up environment variable changes."""
    modules_to_reload = [name for name in sys.modules if name.startswith("pdex")]
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])


def test_correctness_comparison():
    """Verify that both experimental modes produce consistent results."""
    PERT_COL = "perturbation"
    CONTROL_VAR = "control"
    RANDOM_SEED = 42
    N_CELLS = 100
    N_GENES = 50
    N_PERTS = 10

    # Create test data
    np.random.seed(RANDOM_SEED)
    obs = pd.DataFrame(
        {
            PERT_COL: np.random.choice(
                [f"pert_{i}" for i in range(N_PERTS)] + [CONTROL_VAR],
                size=N_CELLS,
                replace=True,
            ),
        },
        index=np.arange(N_CELLS).astype(str),
    )

    X = np.random.randint(0, 1000, size=(N_CELLS, N_GENES))
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(N_GENES)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Test both modes
    with pytest.MonkeyPatch().context() as m:
        m.setenv("USE_EXPERIMENTAL", "0")
        _reload_pdex()
        from pdex import parallel_differential_expression

        reference_result = parallel_differential_expression(
            adata,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            metric="wilcoxon",
            num_workers=4,
        )

    with pytest.MonkeyPatch().context() as m:
        m.setenv("USE_EXPERIMENTAL", "1")
        _reload_pdex()
        from pdex import parallel_differential_expression

        experimental_result = parallel_differential_expression(
            adata,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            metric="wilcoxon",
            num_workers=4,
        )

    # Both should return valid results
    assert reference_result is not None
    assert experimental_result is not None
    assert len(reference_result) > 0
    assert len(experimental_result) > 0


@pytest.mark.parametrize(
    "n_cells,n_genes,n_perts",
    [
        (500, 100, 10),
        (1000, 300, 100),
        (2000, 500, 50),
    ],
)
@pytest.mark.parametrize("use_experimental", [True, False])
def test_benchmark_parameterized_datasets(
    benchmark, n_cells, n_genes, n_perts, use_experimental
):
    """Benchmark different dataset sizes with experimental flag toggle."""
    # Constants
    PERT_COL = "perturbation"
    CONTROL_VAR = "control"
    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)

    obs = pd.DataFrame(
        {
            PERT_COL: np.random.choice(
                [f"pert_{i}" for i in range(n_perts)] + [CONTROL_VAR],
                size=n_cells,
                replace=True,
            ),
        },
        index=np.arange(n_cells).astype(str),
    )

    X = np.random.randint(0, 1000, size=(n_cells, n_genes))
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    with pytest.MonkeyPatch().context() as m:
        m.setenv("USE_EXPERIMENTAL", "1" if use_experimental else "0")
        _reload_pdex()
        from pdex import parallel_differential_expression

        result = benchmark(
            parallel_differential_expression,
            adata,
            reference=CONTROL_VAR,
            groupby_key=PERT_COL,
            metric="wilcoxon",
            num_workers=2,
        )

    assert result is not None
    assert len(result) > 0
    assert "p_value" in result.columns
    assert "fold_change" in result.columns
