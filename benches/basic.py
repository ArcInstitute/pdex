import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pdex import (
    parallel_differential_expression,
    parallel_differential_expression_vec,
)

# Test parameters
N_CELLS = 100
N_GENES = 300
N_PERTS = 100
PERT_COL = "perturbation"
CONTROL_VAR = "control"
RANDOM_SEED = 42


@pytest.fixture
def adata():
    """Create a random AnnData object for testing."""
    np.random.seed(RANDOM_SEED)
    
    # Create observations with perturbations
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
    
    # Create gene expression matrix
    X = np.random.randint(0, 1000, size=(N_CELLS, N_GENES))
    
    # Create variable names
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(N_GENES)])
    
    return ad.AnnData(X=X, obs=obs, var=var)


def test_benchmark_reference_implementation(benchmark, adata):
    """Benchmark reference implementation (batch processing)."""
    result = benchmark(
        parallel_differential_expression,
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
        num_workers=4,
    )
    assert result is not None
    assert len(result) > 0


def test_benchmark_fast_implementation(benchmark, adata):
    """Benchmark fast implementation (gene batching)."""
    result = benchmark(
        parallel_differential_expression_vec,
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
    )
    assert result is not None
    assert len(result) > 0


def test_correctness_comparison(adata):
    """Verify that all implementations produce equivalent results."""
    # Run both implementations
    reference_result = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
        num_workers=4,
    )
    
    fast_result = parallel_differential_expression_vec(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
    )
    
    # Sort for comparison
    ref_sorted = reference_result.sort_values(["target", "feature"]).reset_index(drop=True)
    fast_sorted = fast_result.sort_values(["target", "feature"]).reset_index(drop=True)
    
    # Check shapes
    assert ref_sorted.shape == fast_sorted.shape
    
    # Check key columns
    for col in ["target", "reference", "feature"]:
        assert ref_sorted[col].equals(fast_sorted[col])
    
    # Check numeric columns with tolerance
    numeric_cols = [
        "target_mean",
        "reference_mean",
        "percent_change",
        "fold_change",
    ]
    for col in numeric_cols:
        allowed_tolerance = 1e-6 if col != "percent_change" else 0.01
        assert np.allclose(
            ref_sorted[col], fast_sorted[col], atol=allowed_tolerance, equal_nan=True
        ), f"Column '{col}' values differ beyond tolerance {allowed_tolerance}"