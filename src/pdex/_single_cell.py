import logging
import multiprocessing as mp
from collections.abc import Iterator
from functools import partial
from multiprocessing.shared_memory import SharedMemory

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
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
