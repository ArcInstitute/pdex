"""Parallelization helpers for differential expression workflows.

This module encapsulates the reusable pieces required to parallelize the
low-memory chunked implementation. It provides:

- Default parallelization heuristics
- Utilities for configuring the shared numba thread pool
- Helpers for per-target processing of gene chunks
- A simple ThreadPoolExecutor wrapper with progress reporting
- A thin wrapper around the numba-accelerated Wilcoxon ranksum kernel
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.stats import anderson_ksamp, mannwhitneyu, ttest_ind
from tqdm import tqdm

from ._single_cell import prepare_ranksum_buffers, ranksum_kernel_with_pool

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
    if num_workers <= 1:
        iterable = (
            tqdm(targets, desc="Processing targets")
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
                desc="Processing targets",
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
