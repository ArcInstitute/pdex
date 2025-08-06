import time
import numpy as np
import pandas as pd
import anndata as ad
from pdex import (
    parallel_differential_expression,
    parallel_differential_expression_vec,
)

# Test parameters
# N_CELLS = 10_000
# N_GENES = 20_000

# N_CELLS = 300
# N_GENES = 1000

N_CELLS = 100
N_GENES = 500

N_PERTS = 300
PERT_COL = "perturbation"
CONTROL_VAR = "control"
RANDOM_SEED = 42


def build_random_anndata(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_perts: int = N_PERTS,
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Create a random AnnData object for testing."""
    np.random.seed(random_state)

    # Create observations with perturbations
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

    # Create gene expression matrix
    X = np.random.randint(0, 1000, size=(n_cells, n_genes))

    # Create variable names
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])

    return ad.AnnData(X=X, obs=obs, var=var)


def verify_correctness(results_dict: dict) -> bool:
    """Verify that all implementations produce the same results."""
    # Get the first result as reference
    ref_name, ref_df = next(iter(results_dict.items()))
    ref_sorted = ref_df.sort_values(["target", "feature"]).reset_index(drop=True)

    all_match = True
    for name, df in results_dict.items():
        if name == ref_name:
            continue

        # Sort for comparison
        df_sorted = df.sort_values(["target", "feature"]).reset_index(drop=True)

        # Check shapes
        if df_sorted.shape != ref_sorted.shape:
            print(
                f"❌ {name}: Shape mismatch - {df_sorted.shape} vs {ref_sorted.shape}"
            )
            all_match = False
            continue

        # Check key columns
        for col in ["target", "reference", "feature"]:
            if not df_sorted[col].equals(ref_sorted[col]):
                print(f"❌ {name}: Column '{col}' mismatch")
                all_match = False

        # Check numeric columns with tolerance
        numeric_cols = [
            "target_mean",
            "reference_mean",
            "percent_change",
            "fold_change",
        ]
        for col in numeric_cols:
            # TODO: explore relative tolerance on real data
            # if not np.allclose(df_sorted[col], ref_sorted[col], rtol=1e-2, equal_nan=True):

            allowed_tolerance = 1e-6 if col != "percent_change" else 0.01
            if not np.allclose(
                df_sorted[col], ref_sorted[col], atol=allowed_tolerance, equal_nan=True
            ):
                print(f"❌ {name}: Column '{col}' values differ")
                all_match = False
            else:
                print(f"✅ {name}: Column '{col}' values match within {allowed_tolerance} tolerance")

        match_status = "✅" if all_match else "❌"
        print(
            f"{match_status} {name}: Results {'match' if all_match else 'differ from'} reference"
        )

    return all_match


def benchmark_implementations():
    """Benchmark all implementations and verify correctness."""
    print("=" * 60)
    print(
        f"Benchmarking with {N_CELLS} cells, {N_GENES} genes, {N_PERTS} perturbations"
    )
    print("=" * 60)

    # Create test data
    adata = build_random_anndata()

    # Store results for correctness verification
    results = {}
    timings = {}

    # Test reference implementation
    print("\n1. Reference implementation (batch processing):")
    start = time.time()
    results["reference"] = parallel_differential_expression(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
        num_workers=1,
        batch_size=100,
    )
    timings["reference"] = time.time() - start
    print(f"   Time: {timings['reference']:.3f} seconds")

    # Test fast implementation
    print("\n2. fast with 4 workers (gene batching):")
    start = time.time()
    results["fast"] = parallel_differential_expression_vec(
        adata,
        reference=CONTROL_VAR,
        groupby_key=PERT_COL,
        metric="wilcoxon",
    )
    timings["fast"] = time.time() - start
    print(f"   Time: {timings['fast']:.3f} seconds")
    print(f"   Speedup: {timings['reference']/timings['fast']:.1f}x")

    # Verify correctness
    print("\n" + "=" * 60)
    print("Correctness Verification:")
    print("=" * 60)
    verify_correctness(results)

    # Summary table
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    print(f"{'Implementation':<30} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 52)
    for name, timing in timings.items():
        speedup = timings["reference"] / timing
        print(f"{name:<30} {timing:<12.3f} {speedup:<10.1f}x")


if __name__ == "__main__":
    benchmark_implementations()
