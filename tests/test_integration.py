import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import polars as pl

from pdex import parallel_differential_expression

# Test Constants
N_CELLS = 2000
N_GENES = 300
N_PERTS = 10
RANDOM_SEED = 42
CORRELATION_THRESHOLD = 0.999
NUM_WORKERS = 4


def _simulate_single_cell_counts(
    n_cells: int,
    n_genes: int,
    n_perts: int,
    base_mean: float = 100.0,
    dispersion: float = 0.5,
    effect_size: float = 2.0,
    affected_gene_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Simulate negative binomial single-cell counts with perturbation effects.

    Args:
        n_cells: Number of cells to simulate.
        n_genes: Number of genes to simulate.
        n_perts: Number of perturbations (excluding control).
        base_mean: Base mean expression level.
        dispersion: Negative binomial dispersion parameter (n).
        effect_size: Fold-change for affected genes in perturbations.
        affected_gene_fraction: Fraction of genes affected per perturbation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X matrix, obs DataFrame, var DataFrame).
    """
    rng = np.random.default_rng(seed)

    # Generate perturbation labels
    groups = ["control"] + [f"pert_{i}" for i in range(n_perts)]
    obs = pd.DataFrame({"perturbation": rng.choice(groups, size=n_cells)})

    # Generate gene metadata
    var = pd.DataFrame(index=pd.Index([f"gene_{i}" for i in range(n_genes)]))

    # Base expression means for each gene (log-normal distribution)
    gene_means = rng.lognormal(mean=np.log(base_mean), sigma=1.0, size=n_genes)

    # Initialize counts matrix
    X = np.zeros((n_cells, n_genes), dtype=np.int32)

    # Generate counts for each group
    for group in groups:
        mask = obs["perturbation"] == group
        n_group_cells = mask.sum()

        if n_group_cells == 0:
            continue

        # Adjust means for perturbation effects
        group_means = gene_means.copy()

        if group != "control":
            # Select random subset of genes to be affected
            n_affected = int(n_genes * affected_gene_fraction)
            # Use a consistent random subset per group based on seed
            # To do this deterministically per group, we could seed with group hash,
            # but rng is already seeded.
            affected_genes = rng.choice(n_genes, size=n_affected, replace=False)

            # Apply effect size
            group_means[affected_genes] *= effect_size

        # Calculate p for Negative Binomial (n, p) parameterization
        # where mean = n(1-p)/p => p = n / (n + mean)
        # dispersion parameter is 'n' (number of successes)
        r = dispersion
        p = r / (r + group_means)

        # Generate counts
        # Broadcast p to (n_group_cells, n_genes)
        counts = rng.negative_binomial(n=r, p=p, size=(n_group_cells, n_genes))
        X[mask, :] = counts

    return X, obs, var


def _build_integration_anndata(
    n_cells: int = 2000,
    n_genes: int = 300,
    n_perts: int = 10,
    sparse: bool = False,
    log_transform: bool = False,
    seed: int = 42,
) -> ad.AnnData:
    """Build AnnData object for integration testing.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        n_perts: Number of perturbations.
        sparse: If True, store X as CSR sparse matrix.
        log_transform: If True, apply log1p transformation.
        seed: Random seed.

    Returns:
        Configured AnnData object.
    """
    X, obs, var = _simulate_single_cell_counts(
        n_cells=n_cells, n_genes=n_genes, n_perts=n_perts, seed=seed
    )

    # Convert to float for log transform or if sparse (usually float)
    X = X.astype(np.float32)

    if log_transform:
        X = np.log1p(X)

    if sparse:
        X = sp.csr_matrix(X)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def _run_pdex_both_modes(
    adata: ad.AnnData,
    reference: str = "control",
    groupby_key: str = "perturbation",
    num_workers: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """Run pdex with both low_memory modes and merge results.

    Args:
        adata: Input AnnData object.
        reference: Reference group name.
        groupby_key: Column in obs for grouping.
        num_workers: Number of parallel workers.
        **kwargs: Additional arguments passed to parallel_differential_expression.

    Returns:
        Merged DataFrame with _standard and _lowmem suffixes.
    """
    # Run standard mode
    res_standard = parallel_differential_expression(
        adata,
        groupby_key=groupby_key,
        reference=reference,
        num_workers=num_workers,
        low_memory=False,
        **kwargs,
    )
    if isinstance(res_standard, pl.DataFrame):
        res_standard = res_standard.to_pandas()

    # Run low_memory mode
    res_lowmem = parallel_differential_expression(
        adata,
        groupby_key=groupby_key,
        reference=reference,
        num_workers=num_workers,
        low_memory=True,
        **kwargs,
    )
    if isinstance(res_lowmem, pl.DataFrame):
        res_lowmem = res_lowmem.to_pandas()

    # Merge results
    # Columns to join on
    join_cols = ["target", "reference", "feature"]

    # Verify columns exist
    for col in join_cols:
        if col not in res_standard.columns:
            raise KeyError(f"Column {col} missing from standard results")
        if col not in res_lowmem.columns:
            raise KeyError(f"Column {col} missing from low_memory results")

    merged = pd.merge(
        res_standard, res_lowmem, on=join_cols, suffixes=("_standard", "_lowmem")
    )

    return merged


def _assert_high_correlation(
    results: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.9999,
) -> None:
    """Assert Pearson correlation exceeds threshold for all columns.

    Args:
        results: Merged results DataFrame from _run_pdex_both_modes.
        columns: Column names to validate (without suffixes).
        threshold: Minimum acceptable Pearson r value.

    Raises:
        AssertionError: If any column correlation is below threshold.
    """
    for col in columns:
        col_std = f"{col}_standard"
        col_low = f"{col}_lowmem"

        if col_std not in results.columns or col_low not in results.columns:
            raise KeyError(f"Columns {col_std} and/or {col_low} not found in results")

        valid_mask = results[col_std].notna() & results[col_low].notna()

        # If no valid data points
        if not valid_mask.any():
            if bool(results[col_std].isna().all()) and bool(
                results[col_low].isna().all()
            ):
                continue
            else:
                raise AssertionError(f"All valid values are NaN for {col}")

        # Calculate correlation on valid data
        x = results.loc[valid_mask, col_std]
        y = results.loc[valid_mask, col_low]

        # If constant values (std dev is 0), correlation is undefined (NaN)
        if x.std() == 0 and y.std() == 0:
            # If both are constant and equal, that's fine
            if np.allclose(x, y):
                continue

        if len(x) < 2:
            continue

        corr = np.corrcoef(x, y)[0, 1]

        if np.isnan(corr):
            # This happens if one is constant but not the other, or both constant
            # Check for absolute difference
            diff = np.abs(x - y).mean()
            if diff < 1e-6:
                continue
            else:
                raise AssertionError(f"Correlation is NaN and values differ for {col}")

        if corr < threshold:
            print(f"Correlation failure for {col}: r={corr:.6f} < {threshold}")
            # Show sample of mismatches
            print(results.loc[valid_mask, [col_std, col_low]].head())

        assert corr > threshold, (
            f"Correlation for {col} is {corr:.6f}, expected > {threshold}"
        )


def test_integration_dense_counts():
    """Test integration with dense numpy array and integer counts."""
    adata = _build_integration_anndata(
        n_cells=N_CELLS,
        n_genes=N_GENES,
        n_perts=N_PERTS,
        sparse=False,
        log_transform=False,
        seed=RANDOM_SEED,
    )

    results = _run_pdex_both_modes(adata, num_workers=NUM_WORKERS)

    cols_to_validate = [
        "fold_change",
        "p_value",
        "fdr",
        "statistic",
        "target_mean",
        "reference_mean",
    ]

    _assert_high_correlation(
        results, columns=cols_to_validate, threshold=CORRELATION_THRESHOLD
    )


def test_integration_dense_log1p():
    """Test integration with dense numpy array and log1p-transformed values."""
    adata = _build_integration_anndata(
        n_cells=N_CELLS,
        n_genes=N_GENES,
        n_perts=N_PERTS,
        sparse=False,
        log_transform=True,
        seed=RANDOM_SEED,
    )

    results = _run_pdex_both_modes(adata, num_workers=NUM_WORKERS)

    cols_to_validate = [
        "fold_change",
        "p_value",
        "fdr",
        "statistic",
        "target_mean",
        "reference_mean",
    ]

    _assert_high_correlation(
        results, columns=cols_to_validate, threshold=CORRELATION_THRESHOLD
    )


def test_integration_sparse_csr_counts():
    """Test integration with CSR sparse matrix and integer counts."""
    adata = _build_integration_anndata(
        n_cells=N_CELLS,
        n_genes=N_GENES,
        n_perts=N_PERTS,
        sparse=True,
        log_transform=False,
        seed=RANDOM_SEED,
    )

    results = _run_pdex_both_modes(adata, num_workers=NUM_WORKERS)

    cols_to_validate = [
        "fold_change",
        "p_value",
        "fdr",
        "statistic",
        "target_mean",
        "reference_mean",
    ]

    _assert_high_correlation(
        results, columns=cols_to_validate, threshold=CORRELATION_THRESHOLD
    )


def test_integration_sparse_csr_log1p():
    """Test integration with CSR sparse matrix and log1p-transformed values."""
    adata = _build_integration_anndata(
        n_cells=N_CELLS,
        n_genes=N_GENES,
        n_perts=N_PERTS,
        sparse=True,
        log_transform=True,
        seed=RANDOM_SEED,
    )

    results = _run_pdex_both_modes(adata, num_workers=NUM_WORKERS)

    cols_to_validate = [
        "fold_change",
        "p_value",
        "fdr",
        "statistic",
        "target_mean",
        "reference_mean",
    ]

    _assert_high_correlation(
        results, columns=cols_to_validate, threshold=CORRELATION_THRESHOLD
    )


def test_integration_subset_groups():
    """Test integration with explicit groups parameter (subset of perturbations)."""
    adata = _build_integration_anndata(
        n_cells=N_CELLS,
        n_genes=N_GENES,
        n_perts=N_PERTS,
        sparse=False,
        log_transform=True,
        seed=RANDOM_SEED,
    )

    # Select a subset of groups to test
    # "pert_0", "pert_1", "pert_2"
    subset_groups = [f"pert_{i}" for i in range(3)]

    # Run standard mode with subset
    res_standard = parallel_differential_expression(
        adata,
        groupby_key="perturbation",
        reference="control",
        groups=subset_groups,
        num_workers=NUM_WORKERS,
        low_memory=False,
    )
    if isinstance(res_standard, pl.DataFrame):
        res_standard = res_standard.to_pandas()

    # Run low_memory mode with subset
    res_lowmem = parallel_differential_expression(
        adata,
        groupby_key="perturbation",
        reference="control",
        groups=subset_groups,
        num_workers=NUM_WORKERS,
        low_memory=True,
    )
    if isinstance(res_lowmem, pl.DataFrame):
        res_lowmem = res_lowmem.to_pandas()

    # Merge results
    join_cols = ["target", "reference", "feature"]
    merged = pd.merge(
        res_standard, res_lowmem, on=join_cols, suffixes=("_standard", "_lowmem")
    )

    # Validate that we only have the requested groups
    unique_targets = merged["target"].unique()
    assert set(unique_targets) == set(subset_groups)

    cols_to_validate = [
        "fold_change",
        "p_value",
        "fdr",
        "statistic",
        "target_mean",
        "reference_mean",
    ]

    _assert_high_correlation(
        merged, columns=cols_to_validate, threshold=CORRELATION_THRESHOLD
    )


def test_integration_varying_chunk_sizes():
    """Validate results consistent across different gene_chunk_size values."""
    adata = _build_integration_anndata(
        n_cells=N_CELLS,
        n_genes=N_GENES,
        n_perts=N_PERTS,
        sparse=False,
        log_transform=True,
        seed=RANDOM_SEED,
    )

    # Run with default chunk size (usually larger)
    # Note: The default in parallel_differential_expression is 1000
    # Our N_GENES is 300, so default takes it all in one chunk.
    res_default = parallel_differential_expression(
        adata,
        groupby_key="perturbation",
        reference="control",
        num_workers=NUM_WORKERS,
        low_memory=True,
        gene_chunk_size=1000,
    )
    if isinstance(res_default, pl.DataFrame):
        res_default = res_default.to_pandas()

    # Run with small chunk size to force multiple chunks
    # 50 genes per chunk -> 6 chunks for 300 genes
    res_small_chunks = parallel_differential_expression(
        adata,
        groupby_key="perturbation",
        reference="control",
        num_workers=NUM_WORKERS,
        low_memory=True,
        gene_chunk_size=50,
    )
    if isinstance(res_small_chunks, pl.DataFrame):
        res_small_chunks = res_small_chunks.to_pandas()

    # Merge results
    join_cols = ["target", "reference", "feature"]
    merged = pd.merge(
        res_default,
        res_small_chunks,
        on=join_cols,
        suffixes=("_default", "_small"),
    )

    # Validate consistency between chunk sizes
    # They should be identical or extremely close (floating point)
    cols_to_validate = [
        "fold_change",
        "p_value",
        "fdr",
        "statistic",
        "target_mean",
        "reference_mean",
    ]

    for col in cols_to_validate:
        col_def = f"{col}_default"
        col_small = f"{col}_small"

        valid_mask = merged[col_def].notna() & merged[col_small].notna()

        if not valid_mask.any():
            continue

        x = merged.loc[valid_mask, col_def]
        y = merged.loc[valid_mask, col_small]

        # Use allclose for strict equality check since it's the same algorithm
        assert np.allclose(x, y, equal_nan=True), (
            f"Mismatch in {col} between chunk sizes"
        )
