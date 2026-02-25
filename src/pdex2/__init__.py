import logging
import warnings
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

log = logging.getLogger(__name__)

PDEX_MODES = Literal["ref", "all", "on_target"]
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


def _build_group_gene_map(
    obs: pd.DataFrame | Dataset2D,
    groupby: str,
    gene_col: str,
    unique_groups: np.ndarray,
    var_names: pd.Index,
) -> dict[str, int]:
    """Returns a mapping of group name -> gene column index in var_names.

    Raises if gene_col is missing or any group maps to multiple genes.
    Skips groups whose target gene is not in var_names, with a warning.
    """
    if gene_col not in obs.columns:
        raise ValueError(
            f"Missing column: {gene_col}. Available: {', '.join(obs.columns)}"
        )
    if groupby not in obs.columns:
        raise ValueError(
            f"Missing column: {groupby}. Available: {', '.join(obs.columns)}"
        )

    # One-pass: drop NaN gene entries, then get unique (group, gene) pairs
    mapping = (
        pd.DataFrame(obs[[groupby, gene_col]])
        .dropna(subset=[gene_col])
        .drop_duplicates()
    )

    # Detect any group mapped to more than one gene
    counts = mapping.groupby(groupby, sort=False)[gene_col].count()
    multi = counts[counts > 1]
    if not multi.empty:
        details = "; ".join(
            f"'{g}' -> {list(mapping.loc[mapping[groupby] == g, gene_col])}"
            for g in multi.index
        )
        raise ValueError(
            f"{len(multi)} group(s) map to multiple genes in '{gene_col}': {details}"
        )

    # Build lookup restricted to unique_groups
    unique_groups_set = set(unique_groups)
    mapping = mapping[mapping[groupby].isin(unique_groups_set)]
    group_to_gene: dict[str, str] = dict(zip(mapping[groupby], mapping[gene_col]))

    skipped: list[str] = []
    group_gene_map: dict[str, int] = {}
    for group in unique_groups:
        if group not in group_to_gene:
            skipped.append(f"'{group}' (no target gene)")
            continue
        gene_name = group_to_gene[group]
        if gene_name not in var_names:
            skipped.append(f"'{group}' (gene '{gene_name}' not in var_names)")
            continue
        group_gene_map[group] = var_names.get_loc(gene_name)

    if skipped:
        msg = (
            f"Skipping {len(skipped)} group(s) with unresolvable target genes: "
            + ", ".join(skipped)
        )
        log.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)
    return group_gene_map


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
    mask_x: np.ndarray | int,
    mask_y: np.ndarray | int | None = None,
) -> np.ndarray | csr_matrix:
    """Returns the matrix of cells that match the mask."""
    if adata.X is None:
        raise ValueError("AnnData object does not have a matrix.")
    if not mask_y:
        return adata.X[mask_x]  # type: ignore
    else:
        return adata.X[mask_x, mask_y]  # type: ignore


def pdex(
    adata: ad.AnnData,
    groupby: str,
    mode: PDEX_MODES = "ref",
    threads: int = 0,
    **kwargs,
) -> pl.DataFrame:
    log.info(
        "pdex called: mode=%s, groupby=%r, n_obs=%d, n_vars=%d",
        mode,
        groupby,
        adata.n_obs,
        adata.n_vars,
    )

    # Set the global threadpool for numba
    set_numba_threadpool(threads)

    _validate_groupby(adata.obs, groupby)

    if mode == "ref":
        reference = kwargs.pop("reference", DEFAULT_REFERENCE)
        log.info("Reference group: %r", reference)
        return _pdex_ref(adata, groupby=groupby, reference=reference)
    elif mode == "all":
        return _pdex_all(adata, groupby=groupby, **kwargs)
    elif mode == "on_target":
        gene_col = kwargs.pop("gene_col", None)
        if gene_col is None:
            raise ValueError("'gene_col' is required for mode='on_target'")
        reference = kwargs.pop("reference", DEFAULT_REFERENCE)
        log.info("on_target: gene_col=%r, reference=%r", gene_col, reference)
        return _pdex_on_target(
            adata, groupby=groupby, gene_col=gene_col, reference=reference
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _pdex_ref(
    adata: ad.AnnData,
    groupby: str,
    reference: str = DEFAULT_REFERENCE,
) -> pl.DataFrame:
    unique_groups, unique_group_indices = _unique_groups(adata.obs, groupby)
    log.info("Found %d groups (excluding reference)", len(unique_groups) - 1)

    ref_index = _identify_reference_index(unique_groups, reference)
    ref_mask = np.flatnonzero(unique_group_indices == ref_index)
    log.info("Reference %r: %d cells", reference, ref_mask.size)

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
    log.info("Found %d groups for 1-vs-rest comparison", len(unique_groups))

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


def _pdex_on_target(
    adata: ad.AnnData,
    groupby: str,
    gene_col: str,
    reference: str = DEFAULT_REFERENCE,
) -> pl.DataFrame:
    unique_groups, unique_group_indices = _unique_groups(adata.obs, groupby)
    ref_index = _identify_reference_index(unique_groups, reference)
    ref_mask = np.flatnonzero(unique_group_indices == ref_index)
    ref_membership = ref_mask.size
    log.info(
        "on_target: %d groups, reference %r has %d cells",
        len(unique_groups) - 1,
        reference,
        ref_membership,
    )

    group_gene_map = _build_group_gene_map(
        adata.obs, groupby, gene_col, unique_groups, adata.var_names
    )

    rows = []
    for group_idx in tqdm(
        range(len(unique_groups)),
        desc="Running parallel differential expression (on-target)",
    ):
        group_name = unique_groups[group_idx]
        if group_name not in group_gene_map:
            continue
        gene_idx = group_gene_map[group_name]

        group_mask = np.flatnonzero(unique_group_indices == group_idx)

        # Slice single gene column — result shape (n_cells, 1)
        group_col = _isolate_matrix(adata, mask_x=group_mask, mask_y=gene_idx)
        ref_col = _isolate_matrix(adata, mask_x=ref_mask, mask_y=gene_idx)

        # Sparse slices come back as matrices; convert to dense
        if isinstance(group_col, csr_matrix):
            group_col = group_col.toarray()
        if isinstance(ref_col, csr_matrix):
            ref_col = ref_col.toarray()
        group_col = np.asarray(group_col).reshape(-1, 1)
        ref_col = np.asarray(ref_col).reshape(-1, 1)

        group_mean = float(group_col.mean())
        ref_mean = float(ref_col.mean())

        fc = float(fold_change(np.array([group_mean]), np.array([ref_mean]))[0])
        pc = float(percent_change(np.array([group_mean]), np.array([ref_mean]))[0])

        mwu_result = mwu(group_col, ref_col)
        p_value = float(np.clip(np.asarray(mwu_result.pvalue).ravel()[0], 0, 1))
        statistic = float(np.asarray(mwu_result.statistic).ravel()[0])

        rows.append(
            {
                "group": group_name,
                "gene": adata.var_names[gene_idx],
                "group_mean": group_mean,
                "ref_mean": ref_mean,
                "group_membership": group_mask.size,
                "ref_membership": ref_membership,
                "fold_change": fc,
                "percent_change": pc,
                "p_value": p_value,
                "statistic": statistic,
            }
        )

    df = pl.DataFrame(rows)
    fdr = false_discovery_control(df["p_value"].to_numpy())
    return df.with_columns(pl.Series("fdr", fdr))
