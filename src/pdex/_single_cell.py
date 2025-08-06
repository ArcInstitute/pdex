import logging
import multiprocessing as mp
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.shared_memory import SharedMemory

import anndata as ad
import numba
import numpy as np
import pandas as pd
import polars as pl
from numba import njit, prange
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import anderson_ksamp, false_discovery_control, mannwhitneyu, ttest_ind
from tqdm import tqdm

from ._utils import guess_is_log

# Configure logger
tools_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KNOWN_METRICS = ["wilcoxon", "anderson", "t-test"]


def _build_shared_matrix(
    data: np.ndarray | np.matrix | csr_matrix | csc_matrix,
) -> tuple[SharedMemory, tuple[int, int], np.dtype]:
    """Create a shared memory matrix from a numpy array."""
    if isinstance(data, np.matrix):
        data = np.asarray(data)
    elif isinstance(data, csr_matrix) or isinstance(data, csc_matrix):
        data = data.toarray()

    # data should be a numpy array at this point
    assert isinstance(data, np.ndarray)

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
) -> list[dict[str, float]]:
    """Process a batch of target gene and feature combinations.

    This is the function that is parallelized across multiple workers.

    Arguments
    =========
    batch_tasks: list[tuple]
        List of tuples containing target mask, reference mask, variable index,
        target name, reference name, and variable name.
    shm_name: str
        Name of the shared memory object.
    shape: tuple[int, int]
        Shape of the matrix.
    dtype: np.dtype
        Data type of the matrix.
    metric: str
        Metric to use for processing.
    tie_correct: bool = False
        Whether to correct for ties.
    is_log1p: bool = False
        Whether to apply log1p transformation.
    exp_post_agg: bool = True
        Whether to apply exponential post-aggregation.
    clip_value: float | int | None
        Default clip value used when log-fold-changes would be NaN or Inf.
        Ignore clipping if set to None.
        fold_change = (
            1/default_clip_value
            if fold_change == inf
            else default_clip_value
            if fold_change == 0
            else fold_change
        )
    **kwargs: Additional keyword arguments.
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

    Raises
    ------
    ValueError
        If the gene is not found in the dataset.
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

    Exponenentiates and subtracts one if `is_log1p == True`

    User can decide whether to exponentiate before or after aggregation.
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


def parallel_differential_expression(
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
    """Calculate differential expression between groups of cells.

    Parameters
    ----------
    adata: ad.AnnData
        Annotated data matrix containing gene expression data
    groups: list[str], optional
        List of groups to compare, defaults to None which compares all groups
    reference: str, optional
        Reference group to compare against, defaults to "non-targeting"
    groupby_key: str, optional
        Key in `adata.obs` to group by, defaults to "target_gene"
    num_workers: int
        Number of workers to use for parallel processing, defaults to 1
    batch_size: int
        Number of combinations to process in each batch, defaults to 100
    metric: str
        The differential expression metric to use [wilcoxon, anderson, t-test]
    tie_correct: bool
        Whether to perform continuity (tie) correction for wilcoxon ranksum test
    is_log1p: bool, optional
        Specify exactly whether the data is log1p transformed - will use heuristic to check if not provided
        (see `pdex._utils.guess_is_log`).
    exp_post_agg: bool
        Whether to perform exponential post-aggregation for calculating fold change
        (default: perform exponential post-aggregation)
    clip_value: float | int | None
        Value to clip fold change to if it is infinite or NaN (default: 20.0). Set to None to disable clipping.
    as_polars: bool
        return the output dataframe as a polars dataframe
    **kwargs:
        keyword arguments to pass to metric

    Returns
    -------
    pd.DataFrame containing differential expression results for each group and feature
    """
    if metric not in KNOWN_METRICS:
        raise ValueError(f"Unknown metric: {metric} :: Expecting: {KNOWN_METRICS}")

    unique_targets = np.array(adata.obs[groupby_key].unique())
    if groups is not None:
        unique_targets = [
            target
            for target in unique_targets
            if target in groups or target == reference
        ]
    unique_features = np.array(adata.var.index)

    if not is_log1p:
        is_log1p = guess_is_log(adata)

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
    logger.info("Creating shared memory memory matrix for parallel computing")
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

    logger.info("Initializing parallel processing pool")
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
    dataframe["fdr"] = false_discovery_control(dataframe["p_value"].values, method="bh")

    if as_polars:
        return pl.DataFrame(dataframe)

    return dataframe


@njit
def _compute_means_numba(X: np.ndarray, mask: np.ndarray, is_log1p: bool, exp_post_agg: bool) -> np.ndarray:
    """Numba-compiled function to compute means for a masked dataset."""
    X_masked = X[mask, :]
    n_samples, n_genes = X_masked.shape
    means = np.empty(n_genes)
    
    for i in range(n_genes):
        if is_log1p:
            if exp_post_agg:
                means[i] = np.expm1(np.mean(X_masked[:, i]))
            else:
                means[i] = np.mean(np.expm1(X_masked[:, i]))
        else:
            means[i] = np.mean(X_masked[:, i])
    
    return means


@njit
def _compute_fold_change_numba(means_target: np.ndarray, means_ref: np.ndarray, clip_value: float) -> np.ndarray:
    """Numba-compiled function to compute fold changes."""
    n_genes = means_target.shape[0]
    fc = np.empty(n_genes)
    
    for i in range(n_genes):
        if means_ref[i] == 0:
            fc[i] = clip_value if not np.isnan(clip_value) else np.nan
        elif means_target[i] == 0:
            fc[i] = 1.0 / clip_value if not np.isnan(clip_value) else 0.0
        else:
            fc[i] = means_target[i] / means_ref[i]
    
    return fc


@njit
def _compute_percent_change_numba(means_target: np.ndarray, means_ref: np.ndarray) -> np.ndarray:
    """Numba-compiled function to compute percent changes."""
    n_genes = means_target.shape[0]
    pcc = np.empty(n_genes)
    
    for i in range(n_genes):
        if means_ref[i] == 0:
            pcc[i] = np.nan
        else:
            pcc[i] = (means_target[i] - means_ref[i]) / means_ref[i]
    
    return pcc


@njit
def _ranksum_single_gene_numba(x_target: np.ndarray, x_ref: np.ndarray) -> tuple[float, float]:
    """Numba-compiled Wilcoxon rank-sum test for a single gene."""
    n_target = x_target.shape[0]
    n_ref = x_ref.shape[0]
    
    if n_target == 0 or n_ref == 0:
        return 1.0, np.nan
    
    # Combine and rank
    combined = np.concatenate((x_target, x_ref))
    ranks = np.empty_like(combined)
    
    # Simple ranking (could be optimized further)
    sorted_indices = np.argsort(combined)
    for i in range(len(combined)):
        ranks[sorted_indices[i]] = i + 1
    
    # Sum ranks for target group
    rank_sum = np.sum(ranks[:n_target])
    
    # U-statistic
    u_stat = rank_sum - n_target * (n_target + 1) / 2
    
    # Normal approximation
    mu = n_target * n_ref / 2
    sigma = np.sqrt(n_target * n_ref * (n_target + n_ref + 1) / 12)
    
    if sigma == 0:
        return 1.0, u_stat
    
    z_score = (u_stat - mu) / sigma
    # Approximate p-value using normal CDF (simplified)
    # For more accuracy, would need to implement erf function
    p_value = 2 * (1 - 0.5 * (1 + np.tanh(np.abs(z_score) * np.sqrt(2 / np.pi))))
    
    return p_value, u_stat


@njit(parallel=True)
def _process_all_targets_numba(
    X: np.ndarray,
    X_ref: np.ndarray, 
    means_ref: np.ndarray,
    target_masks: np.ndarray,
    target_indices: np.ndarray,
    is_log1p: bool,
    exp_post_agg: bool,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process all targets in parallel using numba."""
    n_targets = target_indices.shape[0]
    n_genes = X.shape[1]
    
    # Pre-allocate output arrays
    target_means_out = np.empty((n_targets, n_genes))
    fold_changes_out = np.empty((n_targets, n_genes))
    percent_changes_out = np.empty((n_targets, n_genes))
    p_values_out = np.empty((n_targets, n_genes))
    statistics_out = np.empty((n_targets, n_genes))
    
    # Process each target in parallel
    for t in prange(n_targets):
        target_idx = target_indices[t]
        
        # Get target mask and data
        target_mask = target_masks[target_idx]
        means_target = _compute_means_numba(X, target_mask, is_log1p, exp_post_agg)
        
        # Compute fold changes and percent changes
        fold_changes = _compute_fold_change_numba(means_target, means_ref, clip_value)
        percent_changes = _compute_percent_change_numba(means_target, means_ref)
        
        # Store results
        target_means_out[t, :] = means_target
        fold_changes_out[t, :] = fold_changes
        percent_changes_out[t, :] = percent_changes
        
        # Compute statistical tests for each gene
        X_target = X[target_mask, :]
        for g in range(n_genes):
            p_val, stat = _ranksum_single_gene_numba(X_target[:, g], X_ref[:, g])
            p_values_out[t, g] = p_val
            statistics_out[t, g] = stat
    
    return target_means_out, fold_changes_out, percent_changes_out, p_values_out, statistics_out


def _vectorized_ranksum_test(
    X_target: np.ndarray, X_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilcoxon rank-sum test across all genes simultaneously."""
    n_target, n_genes = X_target.shape
    n_ref = X_ref.shape[0]

    if n_target == 0 or n_ref == 0:
        return np.ones(n_genes), np.full(n_genes, np.nan)

    # Combine target and reference for each gene
    combined_shape = (n_target + n_ref, n_genes)
    combined = np.empty(combined_shape)
    combined[:n_target] = X_target
    combined[n_target:] = X_ref

    # Vectorized ranking across all genes at once
    ranks = np.empty_like(combined)
    for i in range(n_genes):
        ranks[:, i] = np.argsort(np.argsort(combined[:, i])) + 1

    # Sum ranks for target group across all genes
    rank_sums = np.sum(ranks[:n_target, :], axis=0)

    # Vectorized U-statistic calculation
    u_stats = rank_sums - n_target * (n_target + 1) / 2

    # Vectorized p-value calculation using normal approximation
    mu = n_target * n_ref / 2
    sigma = np.sqrt(n_target * n_ref * (n_target + n_ref + 1) / 12)

    # Handle zero variance case
    z_scores = np.where(sigma > 0, (u_stats - mu) / sigma, 0)

    # Two-tailed p-values using normal approximation
    from scipy.stats import norm

    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    return p_values, u_stats



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
        else:
            fc = np.where(means_ref == 0, np.nan, fc)
            fc = np.where(means_target == 0, 0, fc)

        pcc = np.where(means_ref == 0, np.nan, pcc)

    # Statistical tests across all genes simultaneously
    p_values, statistics = _vectorized_ranksum_test(X_target, X_ref)

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
    unique_targets = np.unique(obs_values)

    if groups is not None:
        mask = np.isin(unique_targets, groups + [reference])
        unique_targets = unique_targets[mask]

    if not is_log1p:
        is_log1p = guess_is_log(adata)

    logger.info(
        f"vectorized processing: {len(unique_targets)} targets, {adata.n_vars} genes"
    )

    # Convert to dense matrix for fastest access
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray().astype(np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    # Get reference data once
    reference_mask = obs_values == reference
    X_ref = X[reference_mask, :]

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

    # Use numba for fast parallel processing
    if num_workers == 1:
        logger.info(f"Processing {len(targets_to_process)} targets sequentially")
        # Sequential processing for comparison
        all_results = []
        for target in tqdm(targets_to_process, desc="Processing targets"):
            target_results = _process_single_target_vectorized(
                target=target,
                reference=reference,
                obs_values=obs_values,
                X=X,
                X_ref=X_ref,
                means_ref=means_ref,
                gene_names=gene_names,
                is_log1p=is_log1p,
                exp_post_agg=exp_post_agg,
                clip_value=clip_value,
            )
            all_results.extend(target_results)
    else:
        # Use numba parallel processing
        logger.info(f"Processing {len(targets_to_process)} targets with numba parallel processing")
        
        # Prepare data for numba
        target_names_filtered = np.array(targets_to_process)
        n_targets = len(target_names_filtered)
        
        # Create target masks array and indices
        target_masks_dict = {}
        target_indices = []
        
        for i, target in enumerate(target_names_filtered):
            mask = obs_values == target
            target_masks_dict[i] = mask
            target_indices.append(i)
        
        # Convert to arrays that numba can handle
        target_masks_array = np.array([target_masks_dict[i] for i in target_indices])
        target_indices_array = np.array(target_indices)
        
        # Handle clip_value for numba
        clip_val = clip_value if clip_value is not None else np.nan
        
        # Process all targets in parallel with numba
        (target_means_out, fold_changes_out, percent_changes_out, 
         p_values_out, statistics_out) = _process_all_targets_numba(
            X, X_ref, means_ref, target_masks_array, target_indices_array,
            is_log1p, exp_post_agg, clip_val
        )
        
        # Convert results back to list format
        all_results = []
        for t in range(n_targets):
            target_name = target_names_filtered[t]
            for g in range(len(gene_names)):
                all_results.append({
                    "target": target_name,
                    "reference": reference,
                    "feature": gene_names[g],
                    "target_mean": target_means_out[t, g],
                    "reference_mean": means_ref[g],
                    "percent_change": percent_changes_out[t, g],
                    "fold_change": fold_changes_out[t, g],
                    "p_value": p_values_out[t, g],
                    "statistic": statistics_out[t, g],
                })

    # Create dataframe
    dataframe = pd.DataFrame(all_results)
    dataframe["fdr"] = false_discovery_control(dataframe["p_value"].values, method="bh")

    if as_polars:
        return pl.DataFrame(dataframe)

    return dataframe
