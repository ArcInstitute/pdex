import logging
import warnings
from typing import Any, Literal, cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from anndata.experimental.backed import Dataset2D
from numba_mwu import sparse_column_index
from scipy.sparse import csr_matrix, issparse
from scipy.stats import false_discovery_control
from tqdm import tqdm

from pdex._math import (
    bulk_matrix_arithmetic,
    cpm_bulk,
    log2_fold_change,
    mwu,
    percent_change,
    pseudobulk,
)

from ._utils import _detect_is_log1p, set_numba_threadpool

log = logging.getLogger(__name__)

# Emit warnings and above to stderr by default so auto-detection messages are
# always visible without any logging configuration by the caller.
_handler = logging.StreamHandler()
_handler.setLevel(logging.WARNING)
_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.WARNING)

PDEX_MODES = Literal["ref", "all", "on_target"]
DEFAULT_REFERENCE = "non-targeting"

__all__ = ["pdex", "DEFAULT_REFERENCE", "PDEX_MODES"]


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
    control: str,
    var_names: pd.Index,
) -> dict[str, int]:
    """Returns a mapping of group name -> gene column index in var_names.

    Raises if gene_col is missing or any non-control group maps to multiple genes.
    Logs a warning and skips groups whose target gene is not in var_names.
    """
    if gene_col not in obs.columns:
        raise ValueError(
            f"Missing column: {gene_col}. Available: {', '.join(obs.columns)}"
        )

    # Unique (group, gene) pairs, dropping NaN gene entries
    mapping = pd.DataFrame(obs[[groupby, gene_col]]).drop_duplicates().dropna()

    # Check for non-control groups mapped to more than one gene, then drop control
    mapping = mapping[mapping[groupby] != control]
    counts = mapping[groupby].value_counts()
    multi = counts[counts > 1]
    if not multi.empty:
        multi_groups = ", ".join(multi.index.values)
        raise ValueError(
            f"Groups map to multiple genes in '{gene_col}': {multi_groups}"
        )

    # Build a fast gene-name -> index lookup
    gene_idx_map = {g: idx for idx, g in enumerate(var_names)}
    missing_var = ~mapping[gene_col].isin(gene_idx_map)
    if np.any(missing_var):
        n_missing = int(np.sum(missing_var))
        missing_detail = ", ".join(
            f"{g} ({gene})"
            for g, gene in zip(
                mapping[groupby][missing_var], mapping[gene_col][missing_var]
            )
        )
        msg = f"Found {n_missing} groups with missing on-target genes in adata.var: {missing_detail}"
        log.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    mapping = mapping[~missing_var].copy()
    mapping["VAR_INDEX"] = mapping[gene_col].map(gene_idx_map)
    return mapping.set_index(groupby)["VAR_INDEX"].to_dict()


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
    """Returns the matrix of cells that match the mask, always in-memory."""
    if adata.X is None:
        raise ValueError("AnnData object does not have a matrix.")
    if mask_y is None:
        result = adata.X[mask_x]  # ty: ignore[not-subscriptable]
    else:
        result = adata.X[mask_x, mask_y]  # ty: ignore[not-subscriptable]

    # Fast path: already in-memory
    if isinstance(result, (np.ndarray, csr_matrix)):
        return result
    # Backed sparse -> csr_matrix
    if issparse(result):
        return csr_matrix(result)
    # Backed dense (h5py.Dataset slice, dask, etc.) -> ndarray
    return np.asarray(result)


def _x_has_negative(x: np.ndarray | csr_matrix | None) -> bool:
    """Best-effort check for negative values in the expression matrix.

    Negative values break the CPM computation used by ``cpm_filter`` (counts
    cannot be negative). In-memory dense/sparse matrices are checked in full;
    backed/other inputs fall back to a bounded sample of the leading rows.
    """
    if x is None:
        return False
    sample = x
    if not isinstance(sample, (np.ndarray, csr_matrix)):
        sample_obj = cast(Any, sample)
        sample = sample_obj[: min(1000, sample_obj.shape[0])]
    if issparse(sample):
        sample = csr_matrix(sample)
        return bool(sample.data.size and (sample.data < 0).any())
    arr = np.asarray(sample)
    return bool(arr.size and (arr < 0).any())


def _per_cell_library_sizes(adata: ad.AnnData, is_log1p: bool) -> np.ndarray:
    """Per-cell total expression over all genes, in natural (count) space.

    Needed by ``on_target`` mode's CPM filter: the single-gene slices don't carry
    the library size, so it is precomputed once over the full matrix (``expm1`` is
    applied first when ``is_log1p``). Returns a 1-D float64 array of length n_obs.
    """
    x = _isolate_matrix(adata, np.arange(adata.n_obs))
    if is_log1p:
        if isinstance(x, csr_matrix):
            x = x.copy()
            np.expm1(x.data, out=x.data)
        else:
            x = np.expm1(np.asarray(x, dtype=np.float64))
    if isinstance(x, csr_matrix):
        return np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    return np.asarray(x, dtype=np.float64).sum(axis=1)


def pdex(
    adata: ad.AnnData,
    groupby: str,
    mode: PDEX_MODES = "ref",
    threads: int = 0,
    is_log1p: bool | None = None,
    geometric_mean: bool = True,
    as_pandas: bool = False,
    epsilon: float = 1e-9,
    cpm_filter: float | None = None,
    **kwargs,
) -> pl.DataFrame | pd.DataFrame:
    """Run parallel differential expression analysis on single-cell data.

    For each group defined by ``groupby``, computes per-gene pseudobulk statistics
    (mean expression, fold change, percent change) and a Mann-Whitney U test against
    a reference, returning FDR-corrected p-values.

    Parameters
    ----------
    adata:
        Annotated data matrix. Expression values are read from ``adata.X``, which
        may be dense or CSR sparse.
    groupby:
        Column in ``adata.obs`` that defines the groups (e.g. guide identity).
        Empty strings and NaN values are excluded.
    mode:
        Comparison strategy:

        - ``"ref"`` — each group vs a single reference group (default
          ``"non-targeting"``; override with ``reference=``).
        - ``"all"`` — each group vs all remaining cells (1-vs-rest).
        - ``"on_target"`` — each group vs the reference, but only at the gene
          targeted by that group. Requires ``gene_col=`` kwarg naming a column
          in ``adata.obs`` that maps each group to its target gene.
    threads:
        Number of Numba threads. ``0`` (default) uses all available CPUs.
    is_log1p:
        Whether ``adata.X`` contains log1p-transformed values.

        - ``True`` — data is log1p-transformed; geometric mean is computed as
          ``expm1(mean(X))``.
        - ``False`` — data is raw/normalised counts; geometric mean is computed
          as ``expm1(mean(log1p(X)))``.
        - ``None`` (default) — auto-detected via a max-value heuristic and a
          log warning is emitted. Pass explicitly to suppress the message.
    geometric_mean:
        If ``True`` (default), the pseudobulk summary is the geometric mean of
        expression values, back-transformed to count space via ``expm1``. The
        exact computation depends on ``is_log1p`` (see above). If ``False``, the
        arithmetic mean of ``adata.X`` is used instead.

        In both cases ``target_mean`` / ``ref_mean`` are always returned in
        **natural (count) space**: when ``geometric_mean=False`` and
        ``is_log1p=True`` the data is back-transformed before averaging
        (``mean(expm1(X))``) so the output is consistent regardless of input
        format.
    as_pandas:
        If ``True``, return a :class:`pandas.DataFrame` instead of a
        :class:`polars.DataFrame`. Requires ``pyarrow``.
    epsilon:
        Pseudocount added to the **native (count-space) means** — the denominator
        (and, for ``log2_fold_change``, the numerator) — before computing
        ``fold_change`` and ``percent_change``. It is never applied to the CPM view
        used by ``cpm_filter``. When ``epsilon > 0``, extreme values from near-zero
        reference means (scRNA-seq sparsity artifact) are dampened toward zero, and
        one-sided zeros become large-but-finite instead of ``±inf``. Has no effect on
        the Mann-Whitney U p-value or FDR. Regardless of ``epsilon``, features
        unexpressed in both groups report ``0.0`` (no change) rather than ``NaN``
        (see Returns).

        Default ``1e-9`` acts as a tiny finite-guard so the output contains no
        ``±inf`` by default. Pass ``epsilon=0.0`` to recover the legacy behaviour
        where one-sided zeros yield ``±inf``. For stronger dampening of the sparsity artifact in scRNA-seq
        CRISPRi/CRISPRa screens, larger values (e.g. ``0.5``) trade fold-change
        fidelity for floor suppression; prefer combining the tiny default with
        ``cpm_filter`` to remove the noise floor outright.

        Must be non-negative. Raises :class:`ValueError` if negative.
    cpm_filter:
        Optional counts-per-million (CPM) floor filter. ``None`` (default) disables
        it. When set to a threshold ``T``, a ``(target, feature)`` row is **dropped**
        from the output when the gene's pooled (bulk) CPM is ``<= T`` in **both** the
        target group and the reference (kept if ``target_cpm > T`` **or**
        ``ref_cpm > T``). The pooled CPM is ``Σcounts_gene / Σcounts_all * 1e6`` per
        group, computed on a separate internal CPM view (counts are recovered via
        ``expm1`` when ``is_log1p``); the reported ``target_mean``/``ref_mean`` stay
        in native count space — the output is never normalised. Because the CPM is a
        ratio it is scale-invariant, so ``T`` means the same regardless of how the
        input was normalised. The drop is independent of the Mann-Whitney U result,
        and **FDR is corrected over the surviving genes only** (the filter changes
        the multiple-testing universe). A negative ``T`` keeps everything. Emits a
        :class:`UserWarning` if the data contains negative values (CPM assumes
        non-negative expression).

        ``T = 5`` is a reasonable starting point, but the optimal threshold is
        dataset-dependent (it tracks the separation between the noise floor and
        genuinely expressed genes); inspect the per-gene CPM distribution of your
        data and tune ``T`` empirically rather than relying on a fixed default.
    **kwargs:
        Mode-specific keyword arguments:

        - ``reference`` (str) — reference group name for ``mode="ref"`` and
          ``mode="on_target"``. Defaults to ``"non-targeting"``.
        - ``gene_col`` (str) — required for ``mode="on_target"``. Names a column
          in ``adata.obs`` mapping each group to its target gene in ``adata.var``.

        Unexpected keyword arguments trigger a :class:`UserWarning`.

    Returns
    -------
    pl.DataFrame | pd.DataFrame
        One row per (group, feature) pair with columns: ``target``, ``feature``,
        ``target_mean``, ``ref_mean``, ``target_membership``, ``ref_membership``,
        ``fold_change``, ``log2_fold_change``, ``percent_change``, ``p_value``,
        ``statistic``, ``fdr``.

        ``target_mean`` and ``ref_mean`` are always in **natural (count) space**.

        ``log2_fold_change`` and ``percent_change`` are derived from the pseudobulk
        means (not from the per-cell MWU test inputs): ``log2_fold_change`` is
        ``log2((target_mean + epsilon) / (ref_mean + epsilon))`` and
        ``percent_change`` is ``(target_mean - ref_mean) / (ref_mean + epsilon)``.

        Features unexpressed in both groups (``target_mean == ref_mean == 0`` with
        ``epsilon == 0``) would evaluate to ``0 / 0``; both ``log2_fold_change``
        and ``percent_change`` define this as ``0.0`` (no change) rather than
        ``NaN``. One-sided zeros still produce ``±inf``.

        ``fold_change`` is a **deprecated** alias for ``log2_fold_change``
        (identical values). It is retained for one release to ease migration
        and will be removed in pdex 0.3.0. New code should read
        ``log2_fold_change`` directly. A :class:`FutureWarning` is emitted
        on every ``pdex(...)`` call.  The MWU ``p_value`` and
        ``statistic`` are computed directly on the per-cell expression vectors.

        For ``mode="ref"``, the reference group itself is excluded from the output.

        For ``mode="on_target"`` each group produces a single row (its target gene
        only).
    """
    log.info(
        "pdex called: mode=%s, groupby=%r, n_obs=%d, n_vars=%d",
        mode,
        groupby,
        adata.n_obs,
        adata.n_vars,
    )

    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    warnings.warn(
        "The `fold_change` column in pdex output is deprecated and will be "
        "removed in pdex 0.3.0. Use `log2_fold_change` instead — it contains "
        "the same values (`log2(target_mean / ref_mean)`).",
        FutureWarning,
        stacklevel=2,
    )

    # Set the global threadpool for numba
    set_numba_threadpool(threads)

    _validate_groupby(adata.obs, groupby)

    # Resolve is_log1p — auto-detect if not specified
    if is_log1p is None:
        is_log1p = _detect_is_log1p(adata.X)
        log.warning(
            "is_log1p not specified; auto-detected %s. "
            "Pass is_log1p explicitly to suppress this message.",
            is_log1p,
        )

    log.info("is_log1p=%s, geometric_mean=%s", is_log1p, geometric_mean)

    if cpm_filter is not None and _x_has_negative(adata.X):  # ty: ignore[invalid-argument-type]
        msg = (
            "cpm_filter is set but adata.X contains negative values; CPM assumes "
            "non-negative expression, so the filter results may be meaningless."
        )
        log.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if mode == "ref":
        reference = kwargs.pop("reference", DEFAULT_REFERENCE)
        if kwargs:
            warnings.warn(
                f"Unexpected keyword arguments for mode='ref' (ignored): {', '.join(kwargs)}",
                UserWarning,
                stacklevel=2,
            )
        log.info("Reference group: %r", reference)
        result = _pdex_ref(
            adata,
            groupby=groupby,
            reference=reference,
            geometric_mean=geometric_mean,
            is_log1p=is_log1p,
            epsilon=epsilon,
            cpm_filter=cpm_filter,
        )
    elif mode == "all":
        if kwargs:
            warnings.warn(
                f"Unexpected keyword arguments for mode='all' (ignored): {', '.join(kwargs)}",
                UserWarning,
                stacklevel=2,
            )
        result = _pdex_all(
            adata,
            groupby=groupby,
            geometric_mean=geometric_mean,
            is_log1p=is_log1p,
            epsilon=epsilon,
            cpm_filter=cpm_filter,
        )
    elif mode == "on_target":
        gene_col = kwargs.pop("gene_col", None)
        if gene_col is None:
            raise ValueError("'gene_col' is required for mode='on_target'")
        reference = kwargs.pop("reference", DEFAULT_REFERENCE)
        if kwargs:
            warnings.warn(
                f"Unexpected keyword arguments for mode='on_target' (ignored): {', '.join(kwargs)}",
                UserWarning,
                stacklevel=2,
            )
        log.info("on_target: gene_col=%r, reference=%r", gene_col, reference)
        result = _pdex_on_target(
            adata,
            groupby=groupby,
            gene_col=gene_col,
            reference=reference,
            geometric_mean=geometric_mean,
            is_log1p=is_log1p,
            epsilon=epsilon,
            cpm_filter=cpm_filter,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if as_pandas:
        return result.to_pandas()
    return result


def _cpm_keep_mask(
    target_matrix: np.ndarray | csr_matrix,
    ref_cpm: np.ndarray,
    is_log1p: bool,
    cpm_filter: float,
) -> np.ndarray:
    """Boolean keep mask: keep a gene iff target OR reference pooled CPM exceeds T.

    ``ref_cpm`` is precomputed by the caller (it is constant within a comparison).
    """
    target_cpm = cpm_bulk(target_matrix, is_log1p)
    return (target_cpm > cpm_filter) | (ref_cpm > cpm_filter)


def _assemble_group_frame(
    *,
    target: str,
    feature_names: np.ndarray | pd.Index,
    target_bulk: np.ndarray,
    ref_bulk: np.ndarray,
    target_membership: int,
    ref_membership: int,
    lfc: np.ndarray,
    pc: np.ndarray,
    pvalue: np.ndarray,
    statistic: np.ndarray,
    keep: np.ndarray | None,
) -> pl.DataFrame:
    """Build one group's result frame, applying ``keep`` and correcting FDR over survivors.

    When ``keep`` is ``None`` no filtering is applied (legacy behaviour). The FDR is
    always computed over the rows that remain, so it reflects the post-filter
    multiple-testing universe. An all-``False`` mask yields a height-0 frame with the
    full schema (safe for :func:`polars.concat`).
    """
    feature = np.asarray(feature_names)
    target_mean = np.asarray(target_bulk).ravel()
    ref_mean = np.asarray(ref_bulk).ravel()
    lfc = np.asarray(lfc).ravel()
    pc = np.asarray(pc).ravel()
    pvalue = np.asarray(pvalue).ravel()
    statistic = np.asarray(statistic).ravel()

    if keep is not None:
        feature = feature[keep]
        target_mean = target_mean[keep]
        ref_mean = ref_mean[keep]
        lfc = lfc[keep]
        pc = pc[keep]
        pvalue = pvalue[keep]
        statistic = statistic[keep]

    fdr = false_discovery_control(pvalue) if pvalue.size else np.empty(0, dtype=float)

    return pl.DataFrame(
        {
            "target": np.full(feature.shape[0], target),
            "feature": feature,
            "target_mean": target_mean,
            "ref_mean": ref_mean,
            "target_membership": np.full(feature.shape[0], target_membership),
            "ref_membership": np.full(feature.shape[0], ref_membership),
            "fold_change": lfc,
            "log2_fold_change": lfc,
            "percent_change": pc,
            "p_value": pvalue,
            "statistic": statistic,
            "fdr": fdr,
        }
    )


def _pdex_ref(
    adata: ad.AnnData,
    groupby: str,
    reference: str = DEFAULT_REFERENCE,
    geometric_mean: bool = True,
    is_log1p: bool = False,
    epsilon: float = 0.0,
    cpm_filter: float | None = None,
) -> pl.DataFrame:
    unique_groups, unique_group_indices = _unique_groups(adata.obs, groupby)
    log.info("Found %d groups (excluding reference)", len(unique_groups) - 1)

    ref_index = _identify_reference_index(unique_groups, reference)
    ref_mask = np.flatnonzero(unique_group_indices == ref_index)
    log.info("Reference %r: %d cells", reference, ref_mask.size)

    ref_matrix = _isolate_matrix(adata, ref_mask)
    ref_bulk = pseudobulk(ref_matrix, geometric_mean=geometric_mean, is_log1p=is_log1p)
    ref_membership = ref_mask.size

    # Reference pooled CPM is constant across target groups (CPM view, filter only)
    ref_cpm = cpm_bulk(ref_matrix, is_log1p) if cpm_filter is not None else None

    # Either sparse_column_index or ref_matrix
    ref_data = (
        sparse_column_index(ref_matrix)
        if isinstance(ref_matrix, csr_matrix)
        else ref_matrix
    )

    feature_names = adata.var_names

    results = []
    for group_idx in tqdm(
        range(len(unique_groups)),
        desc="Running parallel differential expression (against reference)",
    ):
        if group_idx == ref_index:
            continue
        group_name = unique_groups[group_idx]
        group_mask = np.flatnonzero(unique_group_indices == group_idx)
        group_matrix = _isolate_matrix(adata, group_mask)
        group_bulk = pseudobulk(
            group_matrix, geometric_mean=geometric_mean, is_log1p=is_log1p
        )

        lfc = log2_fold_change(group_bulk, ref_bulk, epsilon)
        pc = percent_change(group_bulk, ref_bulk, epsilon)
        mwu_result = mwu(group_matrix, ref_data)

        mwu_statistic = mwu_result.statistic
        mwu_pvalue = np.asarray(mwu_result.pvalue).clip(0, 1)

        if cpm_filter is None:
            keep = None
        else:
            assert ref_cpm is not None  # set whenever cpm_filter is not None
            keep = _cpm_keep_mask(group_matrix, ref_cpm, is_log1p, cpm_filter)

        results.append(
            _assemble_group_frame(
                target=group_name,
                feature_names=feature_names,
                target_bulk=group_bulk,
                ref_bulk=ref_bulk,
                target_membership=group_mask.size,
                ref_membership=ref_membership,
                lfc=lfc,
                pc=pc,
                pvalue=mwu_pvalue,
                statistic=mwu_statistic,
                keep=keep,
            )
        )
    return pl.concat(results)


def _pdex_all(
    adata: ad.AnnData,
    groupby: str,
    geometric_mean: bool = True,
    is_log1p: bool = False,
    epsilon: float = 0.0,
    cpm_filter: float | None = None,
) -> pl.DataFrame:
    unique_groups, unique_group_indices = _unique_groups(adata.obs, groupby)
    log.info("Found %d groups for 1-vs-rest comparison", len(unique_groups))

    feature_names = adata.var_names

    results = []
    for group_idx in tqdm(
        range(len(unique_groups)),
        desc="Running parallel differential expression (1 vs Rest)",
    ):
        group_name = unique_groups[group_idx]

        group_mask = np.flatnonzero(unique_group_indices == group_idx)
        rest_mask = np.flatnonzero(
            (unique_group_indices != group_idx) & (unique_group_indices >= 0)
        )

        group_matrix = _isolate_matrix(adata, group_mask)
        rest_matrix = _isolate_matrix(adata, rest_mask)

        group_bulk = pseudobulk(
            group_matrix, geometric_mean=geometric_mean, is_log1p=is_log1p
        )
        rest_bulk = pseudobulk(
            rest_matrix, geometric_mean=geometric_mean, is_log1p=is_log1p
        )

        lfc = log2_fold_change(group_bulk, rest_bulk, epsilon)
        pc = percent_change(group_bulk, rest_bulk, epsilon)
        mwu_result = mwu(group_matrix, rest_matrix)

        mwu_statistic = mwu_result.statistic
        mwu_pvalue = np.asarray(mwu_result.pvalue).clip(0, 1)

        if cpm_filter is None:
            keep = None
        else:
            rest_cpm = cpm_bulk(rest_matrix, is_log1p)
            keep = _cpm_keep_mask(group_matrix, rest_cpm, is_log1p, cpm_filter)

        results.append(
            _assemble_group_frame(
                target=group_name,
                feature_names=feature_names,
                target_bulk=group_bulk,
                ref_bulk=rest_bulk,
                target_membership=group_mask.size,
                ref_membership=rest_mask.size,
                lfc=lfc,
                pc=pc,
                pvalue=mwu_pvalue,
                statistic=mwu_statistic,
                keep=keep,
            )
        )

    return pl.concat(results)


def _pdex_on_target(
    adata: ad.AnnData,
    groupby: str,
    gene_col: str,
    reference: str = DEFAULT_REFERENCE,
    geometric_mean: bool = True,
    is_log1p: bool = False,
    epsilon: float = 0.0,
    cpm_filter: float | None = None,
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
        adata.obs, groupby, gene_col, reference, adata.var_names
    )
    log.info(
        "on_target: evaluating expression of %d group/gene pairs",
        len(group_gene_map),
    )

    # Per-cell library sizes for the CPM filter (single-gene slices lack them)
    lib_cell = (
        _per_cell_library_sizes(adata, is_log1p) if cpm_filter is not None else None
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

        # CPM filter: drop the row iff the target gene is <= T in both sides.
        # Uses arithmetic-mean pooled CPM (consistent with cpm_bulk), independent
        # of the geometric/arithmetic choice for the reported means.
        if cpm_filter is not None:
            assert lib_cell is not None  # set whenever cpm_filter is not None
            target_arith = float(bulk_matrix_arithmetic(group_col, is_log1p)[0])
            ref_arith = float(bulk_matrix_arithmetic(ref_col, is_log1p)[0])
            t_lib = float(lib_cell[group_mask].mean()) if group_mask.size else 0.0
            r_lib = float(lib_cell[ref_mask].mean()) if ref_mask.size else 0.0
            target_cpm = target_arith / t_lib * 1e6 if t_lib > 0 else 0.0
            ref_cpm = ref_arith / r_lib * 1e6 if r_lib > 0 else 0.0
            if not (target_cpm > cpm_filter or ref_cpm > cpm_filter):
                continue

        target_mean = float(
            pseudobulk(group_col, geometric_mean=geometric_mean, is_log1p=is_log1p)[0]
        )
        ref_mean = float(
            pseudobulk(ref_col, geometric_mean=geometric_mean, is_log1p=is_log1p)[0]
        )

        lfc = float(
            log2_fold_change(np.array([target_mean]), np.array([ref_mean]), epsilon)[0]
        )
        pc = float(
            percent_change(np.array([target_mean]), np.array([ref_mean]), epsilon)[0]
        )

        mwu_result = mwu(group_col, ref_col)
        p_value = float(np.clip(np.asarray(mwu_result.pvalue).ravel()[0], 0, 1))
        statistic = float(np.asarray(mwu_result.statistic).ravel()[0])

        rows.append(
            {
                "target": group_name,
                "feature": adata.var_names[gene_idx],
                "target_mean": target_mean,
                "ref_mean": ref_mean,
                "target_membership": group_mask.size,
                "ref_membership": ref_membership,
                "fold_change": lfc,
                "log2_fold_change": lfc,
                "percent_change": pc,
                "p_value": p_value,
                "statistic": statistic,
            }
        )

    if not rows:
        # No surviving rows (e.g. every target gene filtered out, or no group
        # mapped to a gene): return a height-0 frame with the full schema.
        return pl.DataFrame(
            schema={
                "target": pl.Utf8,
                "feature": pl.Utf8,
                "target_mean": pl.Float64,
                "ref_mean": pl.Float64,
                "target_membership": pl.Int64,
                "ref_membership": pl.Int64,
                "fold_change": pl.Float64,
                "log2_fold_change": pl.Float64,
                "percent_change": pl.Float64,
                "p_value": pl.Float64,
                "statistic": pl.Float64,
                "fdr": pl.Float64,
            }
        )

    df = pl.DataFrame(rows)
    fdr = false_discovery_control(df["p_value"].to_numpy())
    return df.with_columns(pl.Series("fdr", fdr))
