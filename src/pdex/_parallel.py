"""Parallelization helpers for differential expression workflows.

This module encapsulates the reusable pieces required to parallelize the
low-memory chunked implementation. It provides:

- Default parallelization heuristics (`get_default_parallelization`)
- Utilities for configuring the shared numba thread pool (`set_numba_threads`)
- Helpers for per-target processing of gene chunks (`process_target_in_chunk`)
- A ThreadPoolExecutor wrapper with consistent progress reporting
  (`process_targets_parallel`)
- A thin wrapper around the numba-accelerated Wilcoxon ranksum kernel
  (`vectorized_ranksum_test`)

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

    means_target = _compute_means(X_target, is_log1p=is_log1p, exp_post_agg=exp_post_agg)
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
        iterable = (
            tqdm(targets, desc=progress_label)
            if show_progress
            else targets
        )
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


def vectorized_ranksum_test(
    X_target: np.ndarray,
    X_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the numba-accelerated Wilcoxon ranksum test across genes."""
    K_cols, pool_cnt, pool_cnt_t = prepare_ranksum_buffers(X_target, X_ref)
    Xt = np.ascontiguousarray(X_target)
    Xr = np.ascontiguousarray(X_ref)
    return ranksum_kernel_with_pool(Xt, Xr, K_cols, pool_cnt, pool_cnt_t)


def prepare_ranksum_buffers(X_target: np.ndarray, X_ref: np.ndarray):
    """Allocate per-thread buffers used by the numba ranksum kernel."""
    K_cols = np.maximum(X_target.max(axis=0), X_ref.max(axis=0)).astype(np.int64)
    K_max = int(K_cols.max())
    Kp1 = K_max + 1

    nthreads = get_num_threads()
    pool_cnt = np.zeros((nthreads, Kp1), dtype=np.int64)
    pool_cnt_t = np.zeros((nthreads, Kp1), dtype=np.int64)
    return K_cols, pool_cnt, pool_cnt_t


@njit(parallel=True, fastmath=True)
def ranksum_kernel_with_pool(X_target, X_ref, K_cols, pool_cnt, pool_cnt_t):
    """Vectorized Wilcoxon ranksum test implemented in numba."""
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

        for i in range(n_t):
            v = int(X_target[i, j])
            cnt[v] += 1
            cnt_t[v] += 1
        for i in range(n_r):
            v = int(X_ref[i, j])
            cnt[v] += 1

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

        u = rank_sum_target - 0.5 * n_t * (n_t + 1)
        u_stats[j] = u

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

        for v in range(Kp1_use):
            cnt[v] = 0
            cnt_t[v] = 0

    return p_values, u_stats


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
