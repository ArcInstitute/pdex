from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from anndata.experimental.backed import Dataset2D
from numba_mwu import mannwhitneyu_columns, mannwhitneyu_sparse
from scipy.sparse import csr_matrix
from scipy.stats import false_discovery_control
from tqdm import tqdm

from pdex2._math import fold_change, percent_change

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

    ntc_index = _identify_reference_index(unique_groups, reference)
    ntc_mask = np.flatnonzero(unique_group_indices == ntc_index)

    ntc_matrix: np.ndarray | csr_matrix = adata.X[ntc_mask]  # type: ignore
    ntc_bulk = ntc_matrix.mean(axis=0)
    ntc_membership = ntc_mask.size

    results = []
    for group_idx in tqdm(
        range(len(unique_groups)),
        desc="Running parallel differential expression (against reference)",
    ):
        group_name = unique_groups[group_idx]
        group_mask = np.flatnonzero(unique_group_indices == group_idx)
        group_matrix: np.ndarray | csr_matrix = adata.X[group_mask]  # type: ignore
        group_bulk = group_matrix.mean(axis=0)

        if isinstance(group_matrix, csr_matrix):
            mwu_results = mannwhitneyu_sparse(group_matrix, ntc_matrix)
        else:
            mwu_results = mannwhitneyu_columns(group_matrix, ntc_matrix)
        fc = fold_change(np.asarray(group_bulk).ravel(), np.asarray(ntc_bulk).ravel())
        pc = percent_change(
            np.asarray(group_bulk).ravel(), np.asarray(ntc_bulk).ravel()
        )

        mwu_statistic = mwu_results.statistic
        mwu_pvalue = np.asarray(mwu_results.pvalue).clip(0, 1)
        mwu_fdr = false_discovery_control(mwu_pvalue)

        results.append(
            pl.DataFrame(
                {
                    "group": group_name,
                    "group_mean": np.asarray(group_bulk).ravel(),
                    "ref_mean": np.asarray(ntc_bulk).ravel(),
                    "group_membership": group_mask.size,
                    "ref_membership": ntc_membership,
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
    return pl.DataFrame([])
