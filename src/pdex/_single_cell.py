import logging
import multiprocessing as mp
import os
from collections.abc import Iterator
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.stats import anderson_ksamp, false_discovery_control, mannwhitneyu, ttest_ind
from tqdm import tqdm

from ._parallel import (
    get_default_parallelization,
    process_target_in_chunk,
    process_targets_parallel,
    should_use_numba,
    set_numba_threads,
    vectorized_ranksum_test,
)
from ._utils import guess_is_log

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_experimental = (os.getenv("USE_EXPERIMENTAL", "0") == "1") or (
    os.getenv("USE_EXPERIMENTAL", "0") == "true"
)

KNOWN_METRICS = ["wilcoxon", "anderson", "t-test"]


def _is_backed_array(data: Any) -> bool:
    """Check if data is a backed/HDF5 array type.

    Args:
        data: The data object to check.

    Returns:
        True if data appears to be a backed/HDF5 array.
    """
    type_name = type(data).__name__
    module_name = type(data).__module__

    # Common backed array types from h5py and anndata
    backed_indicators = [
        "h5py" in module_name,
        "Dataset" in type_name,
        "AnnDataFileManager" in type_name,
        "SparseDataset" in type_name,
        # h5py Dataset has 'id' and 'file' attributes
        hasattr(data, "id") and hasattr(data, "file"),
    ]
    return any(backed_indicators)


def _build_shared_matrix(
    data: np.ndarray | np.matrix | csr_matrix | csc_matrix,
) -> tuple[SharedMemory, tuple[int, int], np.dtype]:
    """Create a shared memory matrix from a numpy array.

    Args:
        data: Expression matrix as numpy array, matrix, or scipy sparse matrix.

    Returns:
        Tuple of (SharedMemory object, shape, dtype).

    Raises:
        TypeError: If data is a backed/HDF5 array (from backed AnnData).
    """
    # Check for backed arrays first and provide a helpful error message
    if _is_backed_array(data):
        raise TypeError(
            "Cannot create shared memory from a backed AnnData object. "
            "The expression matrix (adata.X) is lazily loaded from disk (HDF5). "
            "To fix this, either:\n"
            "  1. Load the AnnData fully into memory: adata = sc.read_h5ad(path) "
            "(without backed='r')\n"
            "  2. Convert to in-memory: adata = adata.to_memory()\n"
            "  3. Slice the data first: adata = adata[subset, :].copy()"
        )

    if isinstance(data, np.matrix):
        data = np.asarray(data)
    elif isinstance(data, csr_matrix) or isinstance(data, csc_matrix):
        data = data.toarray()

    # Validate data type after conversion
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            "Expected numpy array, numpy matrix, or scipy sparse matrix (csr/csc)."
        )

    shared_matrix = SharedMemory(create=True, size=data.nbytes)
    matrix = np.ndarray(data.shape, dtype=data.dtype, buffer=shared_matrix.buf)
    matrix[:] = data
    return shared_matrix, data.shape, data.dtype


def _conclude_shared_memory(shared_memory: SharedMemory):
    """Close and unlink a shared memory."""
    shared_memory.close()
    shared_memory.unlink()


def _combinations_generator(
    target_masks: dict[str, np.ndarray],
    var_indices: dict[str, int],
    reference: str,
    target_list: list[str] | np.ndarray,
    feature_list: list[str] | np.ndarray,
) -> Iterator[tuple]:
    """Generate all combinations of target genes and features."""
    for target in target_list:
        for feature in feature_list:
            yield (
                target_masks[target],
                target_masks[reference],
                var_indices[feature],
                target,
                reference,
                feature,
            )


def _batch_generator(
    combinations: Iterator[tuple],
    batch_size: int,
    num_combinations: int,
) -> Iterator[list[tuple]]:
    """Generate batches of combinations."""
    for _i in range(0, num_combinations, batch_size):
        subset = []
        for _ in range(batch_size):
            try:
                subset.append(next(combinations))
            except StopIteration:
                break
        yield subset


def _process_target_batch_shm(
    batch_tasks: list[tuple],
    shm_name: str,
    shape: tuple[int, int],
    dtype: np.dtype,
    metric: str,
    tie_correct: bool = False,
    is_log1p: bool = False,
    exp_post_agg: bool = True,
    clip_value: float | int | None = 20,
    **kwargs,
) -> list[dict[str, str | float]]:
    """Process a batch of target gene and feature combinations.

    This is the function that is parallelized across multiple workers.

    Args:
        batch_tasks: List of tuples containing target mask, reference mask,
            variable index, target name, reference name, and variable name.
        shm_name: Name of the shared memory object.
        shape: Shape of the matrix.
        dtype: Data type of the matrix.
        metric: Metric to use for processing.
        tie_correct: Whether to correct for ties.
        is_log1p: Whether to apply log1p transformation.
        exp_post_agg: Whether to apply exponential post-aggregation.
        clip_value: Default clip value used when log-fold-changes would be NaN or Inf.
            Ignore clipping if set to None.
        **kwargs: Additional keyword arguments.

    Returns:
        List of result dictionaries for each gene comparison.
    """
    # Open shared memory once for the batch
    existing_shm = SharedMemory(name=shm_name)
    matrix = np.ndarray(shape=shape, dtype=dtype, buffer=existing_shm.buf)

    results = []
    for (
        target_mask,
        reference_mask,
        var_index,
        target_name,
        reference_name,
        var_name,
    ) in batch_tasks:
        if target_name == reference_name:
            continue

        x_tgt = matrix[target_mask, var_index]
        x_ref = matrix[reference_mask, var_index]

        μ_tgt = _sample_mean(x_tgt, is_log1p=is_log1p, exp_post_agg=exp_post_agg)
        μ_ref = _sample_mean(x_ref, is_log1p=is_log1p, exp_post_agg=exp_post_agg)

        fc = _fold_change(μ_tgt, μ_ref, clip_value=clip_value)
        pcc = _percent_change(μ_tgt, μ_ref)

        (pval, stat) = (1.0, np.nan)  # default output in case of failure
        try:
            match metric:
                case "wilcoxon":
                    if tie_correct:
                        # default mannwhitneyu behavior
                        de_result = mannwhitneyu(
                            x_tgt, x_ref, use_continuity=True, **kwargs
                        )
                    else:
                        # equivalent to `ranksums` behavior when `use_continuity=False` but statistic changes
                        de_result = mannwhitneyu(
                            x_tgt, x_ref, use_continuity=False, **kwargs
                        )
                    pval, stat = (de_result.pvalue, de_result.statistic)
                case "anderson":
                    de_result = anderson_ksamp([x_tgt, x_ref], **kwargs)
                    pval, stat = (de_result.pvalue, de_result.statistic)  # type: ignore (has attributes pvalue and statistic)
                case "t-test":
                    de_result = ttest_ind(x_tgt, x_ref, **kwargs)
                    pval, stat = (de_result.pvalue, de_result.statistic)  # type: ignore (has attributes pvalue and statistic)
                case _:
                    raise KeyError(f"Unknown Metric: {metric}")
        except ValueError:
            """Don't bail on runtime value errors - just use default values"""

        results.append(
            {
                "target": target_name,
                "reference": reference_name,
                "feature": var_name,
                "target_mean": μ_tgt,
                "reference_mean": μ_ref,
                "percent_change": pcc,
                "fold_change": fc,
                "p_value": pval,
                "statistic": stat,
            }
        )

    existing_shm.close()
    return results


def _get_obs_mask(
    adata: ad.AnnData,
    target_name: str,
    variable_name: str = "target_gene",
) -> np.ndarray:
    """Return a boolean mask for a specific target name in the obs variable."""
    return adata.obs[variable_name] == target_name


def _get_var_index(
    adata: ad.AnnData,
    target_gene: str,
) -> int:
    """Return the index of a specific gene in the var variable.

    Args:
        adata: Annotated data matrix.
        target_gene: Gene name to find.

    Returns:
        Index of the gene in adata.var.

    Raises:
        ValueError: If the gene is not found in the dataset.
    """
    var_index = np.flatnonzero(adata.var.index == target_gene)
    if len(var_index) == 0:
        raise ValueError(f"Target gene {target_gene} not found in dataset")
    return var_index[0]


def _sample_mean(
    x: np.ndarray,
    is_log1p: bool,
    exp_post_agg: bool,
) -> float:
    """Determine the sample mean of a 1D array.

    Exponentiates and subtracts one if `is_log1p == True`.
    User can decide whether to exponentiate before or after aggregation.

    Args:
        x: Input array.
        is_log1p: Whether data is log1p transformed.
        exp_post_agg: If True, exponentiate after averaging.

    Returns:
        Sample mean.
    """
    if is_log1p:
        if exp_post_agg:
            return np.expm1(np.mean(x))
        else:
            return np.expm1(x).mean()
    else:
        return x.mean()


def _fold_change(
    μ_tgt: float,
    μ_ref: float,
    clip_value: float | int | None = 20,
) -> float:
    """Calculate the fold change between two means."""
    # Return 1 if both means are zero
    if μ_tgt == 0 and μ_ref == 0:
        return np.nan if clip_value is None else 1

    # The fold change is infinite so clip to default value
    if μ_ref == 0:
        return np.nan if clip_value is None else clip_value

    # The fold change is zero so clip to 1 / default value
    if μ_tgt == 0:
        return 0 if clip_value is None else 1 / clip_value

    # Return the fold change
    return μ_tgt / μ_ref


def _percent_change(
    μ_tgt: float,
    μ_ref: float,
) -> float:
    """Calculate the percent change between two means."""
    if μ_ref == 0:
        return np.nan
    return (μ_tgt - μ_ref) / μ_ref


# =============================================================================
# Low-memory chunked implementation for backed AnnData
# =============================================================================


def _load_chunk(X: Any, col_slice: slice) -> np.ndarray:
    """Load a chunk of the matrix, handling various storage types.

    Args:
        X: Matrix (dense, sparse, or backed).
        col_slice: Slice of columns to load.

    Returns:
        Dense numpy array of the chunk.
    """
    chunk = X[:, col_slice]

    # Handle sparse matrices
    if issparse(chunk):
        chunk = chunk.toarray()

    # Handle dask arrays
    if hasattr(chunk, "compute"):
        chunk = chunk.compute()

    # Ensure numpy array with consistent dtype
    return np.asarray(chunk, dtype=np.float32)


def _parallel_differential_expression_chunked(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    gene_chunk_size: int = 1000,
    num_workers: int = 1,
    num_threads: int | None = None,
    metric: str = "wilcoxon",
    tie_correct: bool = True,
    is_log1p: bool | None = None,
    exp_post_agg: bool = True,
    clip_value: float | int | None = 20.0,
    show_progress: bool = True,
    as_polars: bool = False,
    **kwargs,
) -> pd.DataFrame | pl.DataFrame:
    """Low-memory differential expression using gene chunks.

    Processes genes in chunks to minimize peak memory usage. Supports backed
    AnnData where adata.X is an HDF5 dataset on disk.

    Args:
        adata: Annotated data matrix. Can be backed (loaded with backed='r').
        groups: List of groups to compare. Defaults to all non-reference groups.
        reference: Reference group to compare against.
        groupby_key: Key in adata.obs containing group labels.
        gene_chunk_size: Number of genes to process at once.
        num_workers: Number of target-level worker threads. Set to 1 for sequential processing.
        num_threads: Number of numba threads for gene-level parallelization. None lets numba
            auto-detect, while 1 disables numba parallelization.
        metric: Statistical test to use.
        tie_correct: Whether to use tie correction for Wilcoxon test.
        is_log1p: Whether data is log1p transformed. Auto-detected if None.
        exp_post_agg: If is_log1p, whether to exponentiate after averaging.
        clip_value: Clip fold changes to this value when denominator is zero.
        show_progress: Show progress bar.
        as_polars: Return polars DataFrame instead of pandas.
        **kwargs: Additional arguments passed to statistical tests.

    Returns:
        DataFrame with differential expression results.
    """
    # Validate inputs
    if groupby_key not in adata.obs.columns:
        raise KeyError(
            f"Column '{groupby_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    obs_values = np.asarray(adata.obs[groupby_key])
    unique_groups = np.unique(obs_values)

    if reference not in unique_groups:
        raise ValueError(
            f"Reference '{reference}' not found in adata.obs['{groupby_key}']. "
            f"Available groups: {list(unique_groups)}"
        )

    # Determine targets to process
    if groups is not None:
        unique_targets = [t for t in groups if t != reference and t in unique_groups]
        missing = set(groups) - set(unique_groups) - {reference}
        if missing:
            logger.warning(f"Groups not found in data: {missing}")
    else:
        unique_targets = [t for t in unique_groups if t != reference]

    if len(unique_targets) == 0:
        raise ValueError("No target groups to compare after filtering")

    num_workers = 1 if num_workers is None else num_workers
    requested_threads_display = "auto" if num_threads is None else num_threads
    logger.info(
        "Parallelization request: num_workers=%s, num_threads=%s",
        num_workers,
        requested_threads_display,
    )
    actual_num_threads = set_numba_threads(num_threads)
    use_numba = metric == "wilcoxon" and actual_num_threads != 1
    logger.info(
        "Numba threads configured: %s (enabled=%s)",
        actual_num_threads,
        "yes" if use_numba else "no",
    )

    # Auto-detect log1p if needed
    if is_log1p is None:
        # Sample a small chunk to check
        sample_chunk = _load_chunk(adata.X, slice(0, min(100, adata.n_vars)))
        frac = np.modf(sample_chunk.ravel()[:10000])[0]
        is_log1p = bool(np.any(np.abs(frac) > 1e-3))
        if is_log1p:
            logger.info("Auto-detected log1p transformed data.")
        else:
            logger.info("Auto-detected non-log1p data.")

    gene_names = np.asarray(adata.var_names)
    n_genes = len(gene_names)

    # Precompute masks (small memory footprint)
    reference_mask = obs_values == reference
    target_masks = {t: obs_values == t for t in unique_targets}

    n_ref = reference_mask.sum()
    logger.info(
        f"Chunked processing: {len(unique_targets)} targets vs reference ({n_ref} cells), "
        f"{n_genes} genes in chunks of {gene_chunk_size}"
    )

    all_results = []

    # Process genes in chunks
    chunk_iter = range(0, n_genes, gene_chunk_size)
    numba_desc = f"{actual_num_threads}" if use_numba else "off"
    if show_progress:
        chunk_iter = tqdm(
            chunk_iter,
            desc=f"Gene chunks (workers={num_workers}, numba={numba_desc})",
            unit="chunk",
        )

    for chunk_start in chunk_iter:
        chunk_end = min(chunk_start + gene_chunk_size, n_genes)

        # Load chunk from disk/memory
        X_chunk = _load_chunk(adata.X, slice(chunk_start, chunk_end))

        # Extract reference data for this chunk
        X_ref = X_chunk[reference_mask, :]

        # Compute reference means
        if is_log1p and exp_post_agg:
            means_ref = np.expm1(X_ref.mean(axis=0))
        elif is_log1p:
            means_ref = np.expm1(X_ref).mean(axis=0)
        else:
            means_ref = X_ref.mean(axis=0)

        def _process_target(target: str, **_: Any) -> list[dict]:
            target_mask = target_masks[target]
            return process_target_in_chunk(
                target=target,
                reference=reference,
                X_chunk=X_chunk,
                X_ref=X_ref,
                target_mask=target_mask,
                means_ref=means_ref,
                gene_names=gene_names,
                chunk_start=chunk_start,
                metric=metric,
                tie_correct=tie_correct,
                is_log1p=is_log1p,
                exp_post_agg=exp_post_agg,
                clip_value=clip_value,
                use_numba=use_numba,
                **kwargs,
            )

        chunk_results = process_targets_parallel(
            targets=unique_targets,
            process_fn=_process_target,
            num_workers=num_workers,
            show_progress=False,
        )
        all_results.extend(chunk_results)

        # Explicit cleanup to help garbage collector
        del X_chunk, X_ref

    # Build final dataframe
    df = pd.DataFrame(all_results)
    df["p_value"] = df["p_value"].fillna(1.0)

    try:
        df["fdr"] = false_discovery_control(df["p_value"].values, method="bh")
    except ValueError:
        logger.warning("FDR computation failed, using raw p-values")
        df["fdr"] = df["p_value"].copy()

    if as_polars:
        return pl.DataFrame(df)

    return df


# =============================================================================
# Standard shared-memory implementation
# =============================================================================


def _parallel_differential_expression_standard(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    num_workers: int = 1,
    batch_size: int = 100,
    metric: str = "wilcoxon",
    tie_correct: bool = True,
    is_log1p: bool | None = None,
    exp_post_agg: bool = True,
    clip_value: float | int | None = 20.0,
    as_polars: bool = False,
    **kwargs,
) -> pd.DataFrame | pl.DataFrame:
    """Standard shared-memory parallel differential expression.

    This is the original implementation using shared memory for multiprocessing.
    Requires the full matrix to fit in memory.

    Args:
        adata: Annotated data matrix containing gene expression data.
        groups: List of groups to compare, defaults to None which compares all groups.
        reference: Reference group to compare against.
        groupby_key: Key in `adata.obs` to group by.
        num_workers: Number of workers to use for parallel processing.
        batch_size: Number of combinations to process in each batch.
        metric: The differential expression metric to use.
        tie_correct: Whether to perform continuity (tie) correction for wilcoxon.
        is_log1p: Whether data is log1p transformed. Auto-detected if None.
        exp_post_agg: Whether to exponentiate after averaging.
        clip_value: Value to clip fold change to if infinite or zero.
        as_polars: Return polars DataFrame instead of pandas.
        **kwargs: Keyword arguments to pass to metric.

    Returns:
        DataFrame containing differential expression results.
    """
    # Validate inputs
    if groupby_key not in adata.obs.columns:
        raise KeyError(
            f"Column '{groupby_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    unique_targets = np.array(adata.obs[groupby_key].unique())

    if reference not in unique_targets:
        raise ValueError(
            f"Reference '{reference}' not found in adata.obs['{groupby_key}']. "
            f"Available groups: {list(unique_targets)}"
        )

    if groups is not None:
        unique_targets = [
            target
            for target in unique_targets
            if target in groups or target == reference
        ]
    unique_features = np.array(adata.var.index)

    if is_log1p is None:
        is_log1p = guess_is_log(adata)
        if is_log1p:
            logger.info("Auto-detected log1p transformed data.")
        else:
            logger.info("Auto-detected non-log1p data.")
    logger.info("Log1p status: %s", is_log1p)

    # Precompute the number of combinations and batches
    n_combinations = len(unique_targets) * len(unique_features)
    n_batches = n_combinations // batch_size + 1

    # Precompute masks for each target gene
    logger.info("Precomputing masks for each target gene")
    target_masks = {
        target: _get_obs_mask(
            adata=adata, target_name=target, variable_name=groupby_key
        )
        for target in tqdm(unique_targets, desc="Identifying target masks")
    }

    # Precompute variable index for each feature
    logger.info("Precomputing variable indices for each feature")
    var_indices = {
        feature: idx
        for idx, feature in enumerate(
            tqdm(unique_features, desc="Identifying variable indices")
        )
    }

    # Isolate the data matrix from the AnnData object
    logger.info("Creating shared memory matrix for parallel computing")
    (shared_memory, shape, dtype) = _build_shared_matrix(data=adata.X)  # type: ignore
    shm_name = shared_memory.name

    logger.info(f"Creating generator of all combinations: N={n_combinations}")
    combinations = _combinations_generator(
        target_masks=target_masks,
        var_indices=var_indices,
        reference=reference,
        target_list=unique_targets,
        feature_list=unique_features,
    )
    logger.info(f"Creating generator of all batches: N={n_batches}")
    batches = _batch_generator(
        combinations=combinations,
        batch_size=batch_size,
        num_combinations=n_combinations,
    )

    # Partial function for parallel processing
    task_fn = partial(
        _process_target_batch_shm,
        shm_name=shm_name,
        shape=shape,
        dtype=dtype,
        metric=metric,
        tie_correct=tie_correct,
        is_log1p=is_log1p,
        exp_post_agg=exp_post_agg,
        clip_value=clip_value,
        **kwargs,
    )

    logger.info(f"Initializing parallel processing pool with {num_workers} workers")
    with mp.Pool(num_workers) as pool:
        logger.info("Processing batches")
        batch_results = list(
            tqdm(
                pool.imap(task_fn, batches),
                total=n_batches,
                desc="Processing batches",
            )
        )

    # Flatten results
    logger.info("Flattening results")
    results = [result for batch in batch_results for result in batch]

    # Close shared memory
    logger.info("Closing shared memory pool")
    _conclude_shared_memory(shared_memory)

    dataframe = pd.DataFrame(results)
    dataframe["p_value"] = dataframe["p_value"].fillna(
        1.0
    )  # ensure p-values are not NaN ( set to 1.0 )

    try:
        dataframe["fdr"] = false_discovery_control(
            dataframe["p_value"].values, method="bh"
        )
    except ValueError:
        logger.error("Failed to compute FDR - copying p-values")
        dataframe["fdr"] = dataframe["p_value"].copy()

    if as_polars:
        return pl.DataFrame(dataframe)

    return dataframe


# =============================================================================
# Main entry point with automatic backend selection
# =============================================================================


def parallel_differential_expression(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    num_workers: int | None = None,
    batch_size: int = 100,
    num_threads: int | None = None,
    metric: str = "wilcoxon",
    tie_correct: bool = True,
    is_log1p: bool | None = None,
    exp_post_agg: bool = True,
    clip_value: float | int | None = 20.0,
    as_polars: bool = False,
    low_memory: bool | None = None,
    gene_chunk_size: int = 1000,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame | pl.DataFrame:
    """Calculate differential expression between groups of cells.

    This function automatically detects backed AnnData objects and uses an
    appropriate processing strategy. For backed or very large datasets, it
    uses a memory-efficient chunked approach. For in-memory datasets, it uses
    the faster shared-memory parallel approach.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix containing gene expression data. Can be backed
        (loaded with `backed='r'`).
    groups : list[str], optional
        List of groups to compare. Defaults to None which compares all groups.
    reference : str, optional
        Reference group to compare against. Defaults to "non-targeting".
    groupby_key : str, optional
        Key in `adata.obs` to group by. Defaults to "target_gene".
    num_workers : int | None, optional
        Number of workers for parallel processing. In standard mode, defaults to 1 when
        unspecified. In low-memory mode, ``None`` triggers auto-detection via
        :func:`pdex._parallel.get_default_parallelization`.
    batch_size : int, optional
        Number of combinations per batch (standard mode only). Defaults to 100.
    num_threads : int | None, optional
        Number of numba threads to use for gene-level parallelization in low-memory mode.
        ``None`` lets numba auto-detect; ``1`` disables numba parallelization.
    metric : str, optional
        The differential expression metric to use. One of "wilcoxon",
        "anderson", or "t-test". Defaults to "wilcoxon".
    tie_correct : bool, optional
        Whether to perform continuity (tie) correction for wilcoxon ranksum
        test. Defaults to True.
    is_log1p : bool, optional
        Whether data is log1p transformed. Auto-detected if None.
    exp_post_agg : bool, optional
        Whether to exponentiate after averaging for log1p data.
        Defaults to True.
    clip_value : float | int | None, optional
        Value to clip fold change to if infinite or zero. Set to None to
        disable clipping. Defaults to 20.0.
    as_polars : bool, optional
        Return polars DataFrame instead of pandas. Defaults to False.
    low_memory : bool | None, optional
        Force low-memory chunked processing. If None (default), automatically
        uses chunked mode for backed AnnData. Set to True to force chunked
        mode even for in-memory data.
    gene_chunk_size : int, optional
        Number of genes per chunk in low-memory mode. Lower values use less
        memory but are slower. Defaults to 1000.
    show_progress : bool, optional
        Show the low-memory gene chunk progress bar. Defaults to True.
    **kwargs
        Additional keyword arguments passed to the statistical test function.

    Notes
    -----
    The ``num_workers`` and ``num_threads`` parameters only apply to the low-memory
    chunked implementation. When ``low_memory`` is enabled (explicitly or because a
    backed AnnData object was detected), unspecified values fall back to
    :func:`pdex._parallel.get_default_parallelization`. Setting ``num_threads=1``
    disables numba parallelization, while ``num_workers=1`` keeps target processing
    sequential. Both strategies can be combined and share a single numba thread pool.

    The numba acceleration uses dual kernels:
    - Histogram-based kernel for integer count data (O(n + k), faster)
    - Sorting-based kernel for float/normalized data (O(n log n), more general)

    The appropriate kernel is selected automatically based on data type.

    Returns
    -------
    pd.DataFrame | pl.DataFrame
        DataFrame containing differential expression results with columns:
        - target: Target group name
        - reference: Reference group name
        - feature: Gene name
        - target_mean: Mean expression in target group
        - reference_mean: Mean expression in reference group
        - percent_change: Relative change from reference
        - fold_change: Ratio of target to reference means
        - p_value: Statistical test p-value
        - statistic: Test statistic
        - fdr: Benjamini-Hochberg adjusted p-value

    Examples
    --------
    Standard usage with in-memory data:

    >>> results = parallel_differential_expression(
    ...     adata,
    ...     reference="control",
    ...     groupby_key="perturbation",
    ...     num_workers=4,
    ... )

    With backed AnnData (automatically uses chunked mode):

    >>> adata = sc.read_h5ad("large_dataset.h5ad", backed="r")
    >>> results = parallel_differential_expression(
    ...     adata,
    ...     reference="control",
    ...     groupby_key="perturbation",
    ... )

    Force low-memory mode for large in-memory datasets:

    >>> results = parallel_differential_expression(
    ...     adata,
    ...     reference="control",
    ...     low_memory=True,
    ...     gene_chunk_size=500,
    ... )
    """
    if metric not in KNOWN_METRICS:
        raise ValueError(f"Unknown metric: {metric} :: Expecting: {KNOWN_METRICS}")

    # Determine whether to use low-memory mode
    is_backed = _is_backed_array(adata.X)

    if low_memory is None:
        # Auto-detect: use chunked mode for backed data
        use_chunked = is_backed
    else:
        use_chunked = low_memory

    if is_backed and not use_chunked:
        raise TypeError(
            "Cannot use standard mode with backed AnnData. "
            "Either load the data into memory or use low_memory=True."
        )

    if use_chunked:
        if is_backed:
            logger.info("Detected backed AnnData, using low-memory chunked processing")
        else:
            logger.info("Using low-memory chunked processing (low_memory=True)")

        workers = num_workers
        threads = num_threads
        if workers is None or threads is None:
            default_workers, default_threads = get_default_parallelization()
            if workers is None:
                workers = default_workers
            if threads is None:
                threads = default_threads

        if workers is None:
            workers = 1

        requested_numba = threads is None or threads != 1
        if metric != "wilcoxon" and requested_numba:
            logger.warning(
                "Numba parallelization only supports 'wilcoxon' metric. "
                f"Falling back to thread-only parallelization for '{metric}'."
            )

        return _parallel_differential_expression_chunked(
            adata=adata,
            groups=groups,
            reference=reference,
            groupby_key=groupby_key,
            gene_chunk_size=gene_chunk_size,
            num_workers=workers,
            num_threads=threads,
            metric=metric,
            tie_correct=tie_correct,
            is_log1p=is_log1p,
            exp_post_agg=exp_post_agg,
            clip_value=clip_value,
            show_progress=show_progress,
            as_polars=as_polars,
            **kwargs,
        )

    std_workers = num_workers if num_workers is not None else 1
    return _parallel_differential_expression_standard(
        adata=adata,
        groups=groups,
        reference=reference,
        groupby_key=groupby_key,
        num_workers=std_workers,
        batch_size=batch_size,
        metric=metric,
        tie_correct=tie_correct,
        is_log1p=is_log1p,
        exp_post_agg=exp_post_agg,
        clip_value=clip_value,
        as_polars=as_polars,
        **kwargs,
    )


# =============================================================================
# Experimental vectorized implementation (numba-accelerated)
# =============================================================================


def _process_single_target_vectorized(
    target: str,
    reference: str,
    obs_values: np.ndarray,
    X: np.ndarray,
    X_ref: np.ndarray,
    means_ref: np.ndarray,
    gene_names: np.ndarray,
    is_log1p: bool,
    exp_post_agg: bool,
    clip_value: float | int | None,
) -> list[dict]:
    """Process a single target using vectorized operations."""
    if target == reference:
        return []

    # Get target data
    target_mask = obs_values == target
    X_target = X[target_mask, :]

    # Vectorized means calculation
    if is_log1p:
        if exp_post_agg:
            means_target = np.expm1(np.mean(X_target, axis=0))
        else:
            means_target = np.mean(np.expm1(X_target), axis=0)
    else:
        means_target = np.mean(X_target, axis=0)

    # Vectorized fold change and percent change across all genes at once
    with np.errstate(divide="ignore", invalid="ignore"):
        fc = means_target / means_ref
        pcc = (means_target - means_ref) / means_ref

        if clip_value is not None:
            fc = np.where(means_ref == 0, clip_value, fc)
            fc = np.where(means_target == 0, 1 / clip_value, fc)
            fc = np.where((means_ref == 0) & (means_target == 0), 1, fc)
        else:
            fc = np.where(means_ref == 0, np.nan, fc)
            fc = np.where(
                (means_target == 0) & (means_ref != 0),
                0,
                fc,
            )

        pcc = np.where(means_ref == 0, np.nan, pcc)

    # Statistical tests across all genes simultaneously
    p_values, statistics = vectorized_ranksum_test(X_target, X_ref)

    # Build results for all genes at once using vectorized operations
    target_results = [
        {
            "target": target,
            "reference": reference,
            "feature": gene_names[i],
            "target_mean": means_target[i],
            "reference_mean": means_ref[i],
            "percent_change": pcc[i],
            "fold_change": fc[i],
            "p_value": p_values[i],
            "statistic": statistics[i],
        }
        for i in range(len(gene_names))
    ]

    return target_results


def parallel_differential_expression_vec(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    num_workers: int = 1,
    metric: str = "wilcoxon",
    is_log1p: bool | None = None,
    exp_post_agg: bool = True,
    clip_value: float | int | None = 20.0,
    as_polars: bool = False,
) -> pd.DataFrame | pl.DataFrame:
    if metric != "wilcoxon":
        raise ValueError("This implementation currently only supports wilcoxon test")

    # Get unique targets efficiently
    obs_values = adata.obs[groupby_key].values
    unique_targets = np.unique(obs_values)  # type: ignore

    if groups is not None:
        mask = np.isin(unique_targets, groups + [reference])
        unique_targets = unique_targets[mask]

    if is_log1p is None:
        is_log1p = guess_is_log(adata)

    logger.info(
        f"vectorized processing: {len(unique_targets)} targets, {adata.n_vars} genes"
    )

    # Check for backed AnnData before attempting to convert
    if _is_backed_array(adata.X):
        raise TypeError(
            "Cannot process a backed AnnData object. "
            "The expression matrix (adata.X) is lazily loaded from disk (HDF5). "
            "To fix this, either:\n"
            "  1. Load the AnnData fully into memory: adata = sc.read_h5ad(path) "
            "(without backed='r')\n"
            "  2. Convert to in-memory: adata = adata.to_memory()\n"
            "  3. Slice the data first: adata = adata[subset, :].copy()"
        )

    # Convert to dense matrix for fastest access
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray().astype(np.float32)  # type: ignore
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    # Get reference data once
    reference_mask = obs_values == reference
    X_ref = X[reference_mask, :]  # type: ignore

    # Compute reference means once for all genes
    if is_log1p:
        if exp_post_agg:
            means_ref = np.expm1(np.mean(X_ref, axis=0))
        else:
            means_ref = np.mean(np.expm1(X_ref), axis=0)
    else:
        means_ref = np.mean(X_ref, axis=0)

    # Filter out reference target for parallel processing
    targets_to_process = [target for target in unique_targets if target != reference]
    gene_names = adata.var.index.values

    # Process targets sequentially with numba functions
    logger.info(f"Processing {len(targets_to_process)} targets")
    all_results = []
    for target in tqdm(targets_to_process, desc="Processing targets"):
        target_results = _process_single_target_vectorized(
            target=target,
            reference=reference,
            obs_values=obs_values,  # type: ignore
            X=X,
            X_ref=X_ref,
            means_ref=means_ref,
            gene_names=gene_names,  # type: ignore
            is_log1p=is_log1p,
            exp_post_agg=exp_post_agg,
            clip_value=clip_value,
        )
        all_results.extend(target_results)

    # Create dataframe
    dataframe = pd.DataFrame(all_results)
    dataframe["fdr"] = false_discovery_control(dataframe["p_value"].values, method="bh")

    if as_polars:
        return pl.DataFrame(dataframe)

    return dataframe


def parallel_differential_expression_vec_wrapper(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    num_workers: int | None = None,
    batch_size: int = 100,
    num_threads: int | None = None,
    metric: str = "wilcoxon",
    tie_correct: bool = True,
    is_log1p: bool | None = None,
    exp_post_agg: bool = True,
    clip_value: float | int | None = 20.0,
    as_polars: bool = False,
    low_memory: bool | None = None,
    gene_chunk_size: int = 1000,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame | pl.DataFrame:
    ignored_defaults = {
        "num_workers": None,
        "batch_size": 100,
        "num_threads": None,
        "tie_correct": True,
        "low_memory": None,
        "gene_chunk_size": 1000,
        "show_progress": True,
    }
    ignored_values = {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "num_threads": num_threads,
        "tie_correct": tie_correct,
        "low_memory": low_memory,
        "gene_chunk_size": gene_chunk_size,
        "show_progress": show_progress,
    }
    for param, value in ignored_values.items():
        default_value = ignored_defaults[param]
        if value != default_value:
            logger.warning(
                "Experimental vectorized backend ignores parameter '%s'; value %r has no effect.",
                param,
                value,
            )

    if kwargs:
        ignored_keys = ", ".join(sorted(kwargs))
        logger.warning(
            "Experimental vectorized backend ignores additional parameters: %s",
            ignored_keys,
        )

    return parallel_differential_expression_vec(
        adata=adata,
        groups=groups,
        reference=reference,
        groupby_key=groupby_key,
        metric=metric,
        is_log1p=is_log1p,
        exp_post_agg=exp_post_agg,
        clip_value=clip_value,
        as_polars=as_polars,
    )


if use_experimental:
    logger.warning("Using experimental features")
    parallel_differential_expression = parallel_differential_expression_vec_wrapper
