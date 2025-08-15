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


def _create_test_data(n_cells=200, n_genes=100, n_perts=5, seed=42):
    """Helper to create test AnnData."""
    np.random.seed(seed)
    n_per_group = n_cells // (n_perts + 1)

    obs = pd.DataFrame(
        {
            "perturbation": [
                "pert_{}".format(i // n_per_group)
                if i < n_perts * n_per_group
                else "control"
                for i in range(n_cells)
            ]
        }
    )
    X = np.random.poisson(lam=50, size=(n_cells, n_genes)) + 10
    return ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)]),  # type: ignore
    )


def _run_both_modes(adata, reference="control", groupby_key="perturbation"):
    """Run both experimental modes and return results."""
    results = {}
    for use_exp, key in [(False, "ref"), (True, "exp")]:
        with pytest.MonkeyPatch().context() as m:
            m.setenv("USE_EXPERIMENTAL", str(int(use_exp)))
            _reload_pdex()
            from pdex import parallel_differential_expression

            results[key] = parallel_differential_expression(
                adata,
                reference=reference,
                groupby_key=groupby_key,
                metric="wilcoxon",
                num_workers=4,
            )
    return results["ref"], results["exp"]


def test_correctness_comparison():
    """Verify both modes produce consistent results."""
    adata = _create_test_data()
    ref_result, exp_result = _run_both_modes(adata)

    # Check structure matches
    assert ref_result.shape == exp_result.shape
    assert list(ref_result.columns) == list(exp_result.columns)

    # Check means match exactly
    for col in ["reference_mean", "target_mean"]:
        np.testing.assert_allclose(ref_result[col], exp_result[col], rtol=1e-5)

    # Check fold changes match (excluding NaNs)
    mask = ~(ref_result["fold_change"].isna() | exp_result["fold_change"].isna())
    np.testing.assert_allclose(
        ref_result.loc[mask, "fold_change"],
        exp_result.loc[mask, "fold_change"],
        rtol=1e-5,
    )

    # P-values should match within 1e-6 absolute tolerance
    np.testing.assert_allclose(
        ref_result["p_value"], exp_result["p_value"], rtol=1e-2, atol=1e-6
    )


def test_detailed_correctness_metrics():
    """Test with known fold changes."""
    # Create data with specific fold changes
    np.random.seed(42)
    n_cells, n_genes = 200, 100
    n_per_group = n_cells // 6

    obs = pd.DataFrame(
        {"perturbation": ["pert_{}".format(i // n_per_group) for i in range(n_cells)]}
    )
    obs.loc[obs["perturbation"] == "pert_5", "perturbation"] = "control"

    # Create expression with known fold changes
    X = np.zeros((n_cells, n_genes))
    for i, pert in enumerate(obs["perturbation"].unique()):
        mask = obs["perturbation"] == pert
        base = np.random.poisson(lam=100, size=(mask.sum(), n_genes))
        X[mask] = base if pert == "control" else base * (1 + i * 0.5)

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)]),  # type: ignore
    )
    ref_result, exp_result = _run_both_modes(adata)

    # Check fold changes match
    np.testing.assert_allclose(
        ref_result["fold_change"], exp_result["fold_change"], rtol=1e-5
    )

    # P-values should match within 1e-6 absolute tolerance
    np.testing.assert_allclose(
        ref_result["p_value"], exp_result["p_value"], rtol=1e-2, atol=1e-6
    )


def test_edge_cases_numerical_stability():
    """Test extreme values."""
    np.random.seed(42)
    obs = pd.DataFrame({"perturbation": ["pert_1"] * 10 + ["control"] * 10})

    # Test large values
    X_large = np.random.uniform(1e6, 2e6, size=(20, 5))
    adata_large = ad.AnnData(X=X_large, obs=obs)
    ref_large, exp_large = _run_both_modes(adata_large)

    # Means should match (allow small difference for large values)
    np.testing.assert_allclose(
        ref_large["target_mean"], exp_large["target_mean"], rtol=1e-6
    )
    np.testing.assert_allclose(
        ref_large["reference_mean"], exp_large["reference_mean"], rtol=1e-6
    )

    # Test small values
    X_small = np.random.uniform(0.001, 0.01, size=(20, 5))
    adata_small = ad.AnnData(X=X_small, obs=obs)
    ref_small, exp_small = _run_both_modes(adata_small)

    # Fold changes should match
    np.testing.assert_allclose(
        ref_small["fold_change"], exp_small["fold_change"], rtol=1e-5
    )

    # P-values should match within 1e-6 absolute tolerance
    np.testing.assert_allclose(
        ref_small["p_value"], exp_small["p_value"], rtol=1e-2, atol=1e-6
    )

    assert ref_large.shape == exp_large.shape


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
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])  # type: ignore
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
