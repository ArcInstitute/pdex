from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from anndata.experimental.backed import Dataset2D
from numba_mwu import sparse_column_index
from scipy.sparse import csr_matrix
from scipy.stats import false_discovery_control
from tqdm import tqdm

from pdex2._math import bulk_matrix, fold_change, mwu, percent_change

from ._utils import set_numba_threadpool

PDEX_MODES = Literal["ref", "all"]
DEFAULT_REFERENCE = "non-targeting"


def _validate_groupby(obs: pd.DataFrame | Dataset2D, groupby: str):
    """Validates the groupby column exists in the observation data."""
    if groupby not in obs.columns:
        raise ValueError(
            f"Missing column: {groupby}. Available: {', '.join(obs.columns)}"
        )


def _identify_reference_index(unique_groups: np.ndarray, reference: str) -> int:
    """Validates the reference group exists in the reference."""
    ref_idx = np.flatnonzero(unique_groups == reference)
    if len(ref_idx) == 0:
        raise ValueError(
            f"Missing reference: {reference}. Available: {', '.join(unique_groups)}"
        )
    elif len(ref_idx) > 1:
        raise ValueError(
            f"Multiple references found: {reference}. Available: {', '.join(unique_groups)}"
        )
    else:
        return ref_idx[0]


def _unique_groups(
    obs: pd.DataFrame | Dataset2D, groupby: str
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the unique groups in the observation data.

    Removes NaN and empty strings."""
    labels = pd.Categorical(obs[groupby])
    labels = labels.remove_categories(
        [c for c in labels.categories if c == "" or pd.isna(c)]
    )
    groups = np.asarray(labels.categories)
    codes = np.asarray(labels.codes, dtype=np.intp)  # -1 for filtered cells
    return (groups, codes)


def _isolate_matrix(
    adata: ad.AnnData,
    mask: np.ndarray,
) -> np.ndarray | csr_matrix:
    """Returns the matrix of cells that match the mask."""
    if adata.X is None:
        raise ValueError("AnnData object does not have a matrix.")
    return adata.X[mask]  # type: ignore


def pdex(
    adata: ad.AnnData,
    groupby: str,
    mode: PDEX_MODES = "ref",
    threads: int = 0,
    **kwargs,
) -> pl.DataFrame:
    # Set the global threadpool for numba
    set_numba_threadpool(threads)

    _validate_groupby(adata.obs, groupby)

    if mode == "ref":
        return _pdex_ref(
            adata,
            groupby=groupby,
            reference=kwargs.pop("reference", DEFAULT_REFERENCE),
        )
    elif mode == "all":
        return _pdex_all(
            adata,
            groupby=groupby,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _pdex_ref(
    adata: ad.AnnData,
    groupby: str,
    reference: str = DEFAULT_REFERENCE,
) -> pl.DataFrame:
    unique_groups, unique_group_indices = _unique_groups(adata.obs, groupby)

    ref_index = _identify_reference_index(unique_groups, reference)
    ref_mask = np.flatnonzero(unique_group_indices == ref_index)

    ref_matrix = _isolate_matrix(adata, ref_mask)
    ref_bulk = bulk_matrix(ref_matrix)
    ref_membership = ref_mask.size

    # Either sparse_column_index or ref_matrix
    ref_data = (
        sparse_column_index(ref_matrix)
        if isinstance(ref_matrix, csr_matrix)
        else ref_matrix
    )

    results = []
    for group_idx in tqdm(
        range(len(unique_groups)),
        desc="Running parallel differential expression (against reference)",
    ):
        group_name = unique_groups[group_idx]
        group_mask = np.flatnonzero(unique_group_indices == group_idx)
        group_matrix = _isolate_matrix(adata, group_mask)
        group_bulk = bulk_matrix(group_matrix)

        fc = fold_change(group_bulk, ref_bulk)
        pc = percent_change(group_bulk, ref_bulk)
        mwu_result = mwu(group_matrix, ref_data)

        mwu_statistic = mwu_result.statistic
        mwu_pvalue = np.asarray(mwu_result.pvalue).clip(0, 1)
        mwu_fdr = false_discovery_control(mwu_pvalue)

        results.append(
            pl.DataFrame(
                {
                    "group": group_name,
                    "group_mean": np.asarray(group_bulk).ravel(),
                    "ref_mean": np.asarray(ref_bulk).ravel(),
                    "group_membership": group_mask.size,
                    "ref_membership": ref_membership,
                    "fold_change": fc,
                    "percent_change": pc,
                    "p_value": mwu_pvalue,
                    "statistic": mwu_statistic,
                    "fdr": mwu_fdr,
                }
            )
        )
    return pl.concat(results)


def _pdex_all(
    adata: ad.AnnData,
    groupby: str,
    **kwargs,
) -> pl.DataFrame:
    unique_groups, unique_group_indices = _unique_groups(adata.obs, groupby)

    results = []
    for group_idx in tqdm(
        range(len(unique_groups)),
        desc="Running parallel differential expression (1 v Rest",
    ):
        group_name = unique_groups[group_idx]

        group_mask = np.flatnonzero(unique_group_indices == group_idx)
        rest_mask = np.flatnonzero(
            (unique_group_indices != group_idx) & (unique_group_indices >= 0)
        )

        group_matrix = _isolate_matrix(adata, group_mask)
        rest_matrix = _isolate_matrix(adata, rest_mask)

        group_bulk = bulk_matrix(group_matrix)
        rest_bulk = bulk_matrix(rest_matrix)

        fc = fold_change(group_bulk, rest_bulk)
        pc = percent_change(group_bulk, rest_bulk)
        mwu_result = mwu(group_matrix, rest_matrix)

        mwu_statistic = mwu_result.statistic
        mwu_pvalue = np.asarray(mwu_result.pvalue).clip(0, 1)
        mwu_fdr = false_discovery_control(mwu_pvalue)

        results.append(
            pl.DataFrame(
                {
                    "group": group_name,
                    "group_mean": np.asarray(group_bulk).ravel(),
                    "ref_mean": np.asarray(rest_bulk).ravel(),
                    "group_membership": group_mask.size,
                    "ref_membership": rest_mask.size,
                    "fold_change": fc,
                    "percent_change": pc,
                    "p_value": mwu_pvalue,
                    "statistic": mwu_statistic,
                    "fdr": mwu_fdr,
                }
            )
        )

    return pl.concat(results)
