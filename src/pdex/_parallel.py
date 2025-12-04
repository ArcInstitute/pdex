"""Parallelization helpers for differential expression workflows.

This module encapsulates the reusable pieces required to parallelize the
low-memory chunked implementation. It provides:

- Default parallelization heuristics (`get_default_parallelization`)
- Utilities for configuring the shared numba thread pool (`set_numba_threads`)
- Helpers for per-target processing of gene chunks (`process_target_in_chunk`)
- A ThreadPoolExecutor wrapper with consistent progress reporting
  (`process_targets_parallel`)
- Dual numba kernels for Wilcoxon ranksum test:
  - Histogram-based kernel for integer count data (O(n + k))
  - Sorting-based kernel for float data (O(n log n))

These utilities are deliberately decoupled from AnnData-specific logic so
they can be re-used by both the chunked implementation and the experimental
vectorized mode.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads
from scipy.stats import anderson_ksamp, mannwhitneyu, ttest_ind
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "get_default_parallelization",
    "set_numba_threads",
    "is_integer_data",
    "should_use_numba",
    "process_target_in_chunk",
    "process_targets_parallel",
    "vectorized_ranksum_test",
]


def get_default_parallelization() -> tuple[int, int | None]:
    """Return default (num_workers, num_threads) tuple for low-memory mode."""
    cpu_count = os.cpu_count() or 1
    num_workers = max(1, min(4, cpu_count // 4))
    num_threads: int | None = None
    return num_workers, num_threads


def set_numba_threads(num_threads: int | None) -> int:
    """Configure the numba thread pool and report the active thread count."""
    if num_threads is None:
        return get_num_threads()

    set_num_threads(num_threads)
    return num_threads


def is_integer_data(X: np.ndarray, sample_size: int = 10000) -> bool:
    """Return True when all sampled values of X are integer-like.

    Args:
        X: Input array to check.
        sample_size: Maximum number of elements to sample for efficiency.

    Returns:
        True if all sampled values are close to integers.
    """
    flat = np.asarray(X).ravel()
    if flat.size == 0:
        return True

    if flat.size > sample_size:
        rng = np.random.default_rng()
        indices = rng.choice(flat.size, sample_size, replace=False)
        sample = flat[indices]
    else:
        sample = flat

    return np.allclose(sample, np.rint(sample), rtol=0, atol=1e-9)


def should_use_numba(
    X_chunk: np.ndarray,
    metric: str,
    num_threads: int | None,
) -> bool:
    """Determine if numba acceleration should be used for this chunk.

    With dual-kernel support (histogram for integers, sorting for floats),
    numba can be used for any numeric data when metric is wilcoxon.

    Args:
        X_chunk: Data chunk to process.
        metric: Statistical test metric.
        num_threads: Requested numba thread count (1 = disabled).

    Returns:
        True if numba should be used, False otherwise.
    """
    if num_threads == 1:
        return False

    if metric != "wilcoxon":
        return False

    # Supports both integer and float data via dual kernels
    return True


def process_target_in_chunk(
    target: str,
    reference: str,
    X_chunk: np.ndarray,
    X_ref: np.ndarray,
    target_mask: np.ndarray,
    means_ref: np.ndarray,
    gene_names: np.ndarray,
    chunk_start: int,
    metric: str,
    tie_correct: bool,
    is_log1p: bool,
    exp_post_agg: bool,
    clip_value: float | int | None,
    use_numba: bool,
    **kwargs,
) -> list[dict]:
    """Process a single target for the supplied gene chunk."""
    if target == reference:
        return []

    X_target = X_chunk[target_mask, :]
    if X_target.size == 0:
        return []

    means_target = _compute_means(
        X_target, is_log1p=is_log1p, exp_post_agg=exp_post_agg
    )
    fc, pcc = _compute_fold_and_percent_changes(means_target, means_ref, clip_value)

    chunk_size = X_chunk.shape[1]
    effective_numba = use_numba and metric == "wilcoxon"

    if effective_numba:
        p_values, statistics = vectorized_ranksum_test(X_target, X_ref)
    else:
        p_values = np.empty(chunk_size, dtype=np.float64)
        statistics = np.empty(chunk_size, dtype=np.float64)
        for j in range(chunk_size):
            x_tgt = X_target[:, j]
            x_r = X_ref[:, j]
            p_values[j], statistics[j] = _run_metric(
                metric=metric,
                x_target=x_tgt,
                x_reference=x_r,
                tie_correct=tie_correct,
                **kwargs,
            )

    results: list[dict] = []
    for j in range(chunk_size):
        gene_idx = chunk_start + j
        if gene_idx < len(gene_names):
            gene_name = gene_names[gene_idx]
        else:
            # Gene names for the current chunk only
            local_index = j % len(gene_names)
            gene_name = gene_names[local_index]

        results.append(
            {
                "target": target,
                "reference": reference,
                "feature": gene_name,
                "target_mean": float(means_target[j]),
                "reference_mean": float(means_ref[j]),
                "percent_change": float(pcc[j]),
                "fold_change": float(fc[j]),
                "p_value": float(p_values[j]),
                "statistic": float(statistics[j]),
            }
        )

    return results


def process_targets_parallel(
    targets: list[str],
    process_fn: Callable[..., list[dict]],
    num_workers: int,
    show_progress: bool = True,
    **kwargs,
) -> list[dict]:
    """Process the provided targets sequentially or via a thread pool."""
    progress_label = f"Targets (workers={num_workers})"
    if num_workers <= 1:
        iterable = tqdm(targets, desc=progress_label) if show_progress else targets
        results: list[dict] = []
        for target in iterable:
            results.extend(process_fn(target=target, **kwargs))
        return results

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_fn, target=target, **kwargs): target
            for target in targets
        }

        iterator = (
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc=progress_label,
            )
            if show_progress
            else as_completed(futures)
        )
        for future in iterator:
            results.extend(future.result())
    return results


# =============================================================================
# Dual-Kernel Wilcoxon Ranksum Implementation
# =============================================================================


def vectorized_ranksum_test(
    X_target: np.ndarray,
    X_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run numba-accelerated Wilcoxon ranksum test across genes.

    Automatically dispatches to the appropriate kernel:
    - Histogram-based kernel for integer data (O(n + k), faster)
    - Sorting-based kernel for float data (O(n log n), more general)

    Args:
        X_target: Target group expression matrix (n_target_cells × n_genes).
        X_ref: Reference group expression matrix (n_ref_cells × n_genes).

    Returns:
        Tuple of (p_values, u_statistics) arrays of length n_genes.
    """
    # Auto-dispatch based on data type
    if is_integer_data(X_target) and is_integer_data(X_ref):
        return _ranksum_integer(X_target, X_ref)
    else:
        return _ranksum_float(X_target, X_ref)


def _ranksum_integer(
    X_target: np.ndarray,
    X_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-based ranksum for integer count data."""
    K_cols, pool_cnt, pool_cnt_t = _prepare_histogram_buffers(X_target, X_ref)
    Xt = np.ascontiguousarray(X_target)
    Xr = np.ascontiguousarray(X_ref)
    return _ranksum_kernel_histogram(Xt, Xr, K_cols, pool_cnt, pool_cnt_t)


def _ranksum_float(
    X_target: np.ndarray,
    X_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorting-based ranksum for float data."""
    Xt = np.ascontiguousarray(X_target.astype(np.float64))
    Xr = np.ascontiguousarray(X_ref.astype(np.float64))
    return _ranksum_kernel_sorting(Xt, Xr)


def _prepare_histogram_buffers(
    X_target: np.ndarray, X_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate per-thread buffers used by the histogram ranksum kernel."""
    K_cols = np.maximum(X_target.max(axis=0), X_ref.max(axis=0)).astype(np.int64)
    K_max = int(K_cols.max())
    Kp1 = K_max + 1

    nthreads = get_num_threads()
    pool_cnt = np.zeros((nthreads, Kp1), dtype=np.int64)
    pool_cnt_t = np.zeros((nthreads, Kp1), dtype=np.int64)
    return K_cols, pool_cnt, pool_cnt_t


@njit(parallel=True, fastmath=True)
def _ranksum_kernel_histogram(
    X_target, X_ref, K_cols, pool_cnt, pool_cnt_t
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-based Wilcoxon ranksum for integer count data.

    Uses counting sort approach - O(n + k) where k is max value.
    Only valid for non-negative integer data.
    """
    n_t = X_target.shape[0]
    n_r = X_ref.shape[0]
    n_genes = X_target.shape[1]

    p_values = np.empty(n_genes, dtype=np.float64)
    u_stats = np.empty(n_genes, dtype=np.float64)

    for j in prange(n_genes):
        tid = get_thread_id()
        cnt = pool_cnt[tid]
        cnt_t = pool_cnt_t[tid]

        Kp1_use = int(K_cols[j] + 1)

        # Build histograms
        for i in range(n_t):
            v = int(X_target[i, j])
            cnt[v] += 1
            cnt_t[v] += 1
        for i in range(n_r):
            v = int(X_ref[i, j])
            cnt[v] += 1

        # Compute rank sum with tie correction
        running = 1
        rank_sum_target = 0.0
        tie_sum = 0
        for v in range(Kp1_use):
            c = cnt[v]
            if c > 0:
                avg = running + 0.5 * (c - 1)
                rank_sum_target += cnt_t[v] * avg
                tie_sum += c * (c - 1) * (c + 1)
                running += c

        # U statistic
        u = rank_sum_target - 0.5 * n_t * (n_t + 1)
        u_stats[j] = u

        # P-value via normal approximation
        N = n_t + n_r
        if N > 1:
            tie_adj = tie_sum / (N * (N - 1))
            sigma2 = (n_t * n_r) * ((N + 1) - tie_adj) / 12.0
            if sigma2 > 0.0:
                z = (u - 0.5 * n_t * n_r) / math.sqrt(sigma2)
                p_values[j] = math.erfc(abs(z) / math.sqrt(2.0))
            else:
                p_values[j] = 1.0
        else:
            p_values[j] = 1.0

        # Reset buffers for next iteration
        for v in range(Kp1_use):
            cnt[v] = 0
            cnt_t[v] = 0

    return p_values, u_stats


@njit(parallel=True)
def _ranksum_kernel_sorting(X_target, X_ref) -> tuple[np.ndarray, np.ndarray]:
    """Sorting-based Wilcoxon ranksum for float data.

    Uses argsort to compute ranks - O(n log n) per gene.
    Handles ties by averaging ranks.
    """
    n_t = X_target.shape[0]
    n_r = X_ref.shape[0]
    n_genes = X_target.shape[1]
    N = n_t + n_r

    p_values = np.empty(n_genes, dtype=np.float64)
    u_stats = np.empty(n_genes, dtype=np.float64)

    for j in prange(n_genes):
        # Combine target and reference values
        combined = np.empty(N, dtype=np.float64)
        for i in range(n_t):
            combined[i] = X_target[i, j]
        for i in range(n_r):
            combined[n_t + i] = X_ref[i, j]

        # Get sorted indices
        order = np.argsort(combined)

        # Assign ranks with tie handling (average ranks for ties)
        ranks = np.empty(N, dtype=np.float64)
        i = 0
        while i < N:
            # Find the end of the tie group
            tie_start = i
            tie_val = combined[order[i]]
            while i < N and combined[order[i]] == tie_val:
                i += 1
            tie_end = i

            # Average rank for tie group (1-based ranks)
            avg_rank = (tie_start + tie_end + 1) / 2.0

            # Assign average rank to all tied elements
            for k in range(tie_start, tie_end):
                ranks[order[k]] = avg_rank

        # Sum ranks for target group (first n_t elements)
        rank_sum_target = 0.0
        for i in range(n_t):
            rank_sum_target += ranks[i]

        # U statistic
        u = rank_sum_target - 0.5 * n_t * (n_t + 1)
        u_stats[j] = u

        # Compute tie correction factor
        tie_sum = 0.0
        i = 0
        while i < N:
            tie_start = i
            tie_val = combined[order[i]]
            while i < N and combined[order[i]] == tie_val:
                i += 1
            tie_count = i - tie_start
            if tie_count > 1:
                tie_sum += tie_count * (tie_count - 1) * (tie_count + 1)

        # P-value via normal approximation with tie correction
        if N > 1:
            tie_adj = tie_sum / (N * (N - 1))
            sigma2 = (n_t * n_r) * ((N + 1) - tie_adj) / 12.0
            if sigma2 > 0.0:
                z = (u - 0.5 * n_t * n_r) / math.sqrt(sigma2)
                p_values[j] = math.erfc(abs(z) / math.sqrt(2.0))
            else:
                p_values[j] = 1.0
        else:
            p_values[j] = 1.0

    return p_values, u_stats


# =============================================================================
# Legacy API (for backwards compatibility)
# =============================================================================


def prepare_ranksum_buffers(
    X_target: np.ndarray, X_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate per-thread buffers used by the numba ranksum kernel.

    Deprecated: Use vectorized_ranksum_test() which auto-dispatches.
    """
    return _prepare_histogram_buffers(X_target, X_ref)


def ranksum_kernel_with_pool(
    X_target, X_ref, K_cols, pool_cnt, pool_cnt_t
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilcoxon ranksum test implemented in numba.

    Deprecated: Use vectorized_ranksum_test() which auto-dispatches.
    """
    return _ranksum_kernel_histogram(X_target, X_ref, K_cols, pool_cnt, pool_cnt_t)


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_means(
    X: np.ndarray,
    *,
    is_log1p: bool,
    exp_post_agg: bool,
) -> np.ndarray:
    if is_log1p:
        if exp_post_agg:
            return np.expm1(X.mean(axis=0))
        return np.expm1(X).mean(axis=0)
    return X.mean(axis=0)


def _compute_fold_and_percent_changes(
    means_target: np.ndarray,
    means_ref: np.ndarray,
    clip_value: float | int | None,
) -> tuple[np.ndarray, np.ndarray]:
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

    return fc.astype(np.float64), pcc.astype(np.float64)


def _run_metric(
    *,
    metric: str,
    x_target: np.ndarray,
    x_reference: np.ndarray,
    tie_correct: bool,
    **kwargs,
) -> tuple[float, float]:
    (pval, stat) = (1.0, np.nan)
    try:
        match metric:
            case "wilcoxon":
                res = mannwhitneyu(
                    x_target,
                    x_reference,
                    alternative="two-sided",
                    use_continuity=tie_correct,
                    **kwargs,
                )
                pval, stat = res.pvalue, res.statistic
            case "anderson":
                res = anderson_ksamp([x_target, x_reference], **kwargs)
                pval, stat = res.pvalue, res.statistic  # type: ignore[attr-defined]
            case "t-test":
                res = ttest_ind(x_target, x_reference, **kwargs)
                pval, stat = res.pvalue, res.statistic  # type: ignore[attr-defined]
            case _:
                raise ValueError(f"Unknown metric: {metric}")
    except ValueError:
        # Return default values for numerically unstable cases
        logger.debug(
            "Statistical test failed for metric %s; returning defaults",
            metric,
            exc_info=True,
        )
    return float(pval), float(stat)
