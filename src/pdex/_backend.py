"""Normalization layer over ``adata.X`` storage backends.

pdex accepts three flavors of AnnData:

- **in-memory** — ``np.ndarray`` or scipy sparse ``X``
- **backed h5ad** (``ad.read_h5ad(path, backed="r")``) — h5py ``Dataset`` (dense)
  or anndata ``_CSRDataset``/``_CSCDataset`` (sparse) ``X``
- **lazy** (``anndata.experimental.read_lazy(path_or_store)``) — dask-array ``X``
  with dense or scipy-sparse chunks, over local or remote zarr/h5ad stores

Every slice of ``X`` funnels through :func:`realize` so downstream Numba kernels
only ever see ``np.ndarray`` or ``csr_matrix``. Backend detection is duck-typed
(``.compute`` / ``.chunks``) so dask is never imported here — lazy support is an
optional extra (``pdex[lazy]``) and this module stays import-free of it.
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse


def is_lazy(x) -> bool:
    """True when ``x`` is a dask array (i.e. from ``anndata.experimental.read_lazy``)."""
    return type(x).__module__.split(".")[0] == "dask"


def realize(x) -> np.ndarray | csr_matrix:
    """Materialize any backend slice into ``np.ndarray`` or ``csr_matrix``.

    In-memory inputs pass through untouched. Dask arrays are computed first
    (dense chunks yield ``ndarray``, sparse chunks yield scipy sparse); any
    sparse result is normalized to ``csr_matrix``; everything else (h5py/zarr
    dense slices) goes through ``np.asarray``.
    """
    if hasattr(x, "compute"):  # dask
        x = x.compute()
    if isinstance(x, (np.ndarray, csr_matrix)):
        return x
    if issparse(x):
        return csr_matrix(x)
    return np.asarray(x)


def _chunk_size(x, axis: int) -> int | None:
    """Storage chunk extent along ``axis`` when the backend exposes one.

    Dask reports a tuple of block sizes per axis; h5py/zarr report a single
    int per axis (h5py may report ``None`` for contiguous datasets).
    """
    chunks = getattr(x, "chunks", None)
    if chunks is None or len(chunks) <= axis:
        return None
    extent = chunks[axis]
    if isinstance(extent, tuple):  # dask: per-block sizes
        return max(extent) if extent else None
    return int(extent) if extent else None


def default_block_size(
    x, n_other: int, axis: int, target_bytes: int = 256 * 1024**2
) -> int | None:
    """Block extent along ``axis`` for streaming reductions, aligned to storage chunks.

    ``n_other`` is the matrix extent along the other axis. Targets ``target_bytes``
    of dense-equivalent float64 per block (conservative for sparse inputs) and
    rounds to a multiple of the storage chunk extent so a block never reads a
    partial chunk. Returns ``None`` for in-memory ``x`` (no benefit to blocking —
    process everything in one shot).
    """
    if isinstance(x, (np.ndarray, csr_matrix)) or issparse(x):
        return None
    extent = max(1, target_bytes // max(1, n_other * 8))
    chunk = _chunk_size(x, axis=axis) or 1
    return int(max(chunk, (extent // chunk) * chunk))
