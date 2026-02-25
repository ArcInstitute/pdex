import numba as nb
import numpy as np
from numba_mwu import (
    MannWhitneyUResult,
    SparseColumnIndex,
    mannwhitneyu_columns,
    mannwhitneyu_sparse,
)
from scipy.sparse import csr_matrix


@nb.njit(parallel=True)
def _log1p_col_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean of log1p(X) across rows (axis=0) for a dense 2-D array."""
    n_rows, n_cols = matrix.shape
    result = np.zeros(n_cols)
    for j in nb.prange(n_cols):  # type: ignore[attr-defined]
        s = 0.0
        for i in range(n_rows):
            s += np.log1p(matrix[i, j])
        result[j] = s / n_rows
    return result


@nb.njit(parallel=True)
def _expm1_vec(x: np.ndarray) -> np.ndarray:
    """Element-wise expm1 over a 1-D array."""
    result = np.empty_like(x)
    for i in nb.prange(len(x)):  # type: ignore[attr-defined]
        result[i] = np.expm1(x[i])
    return result


@nb.njit(parallel=True)
def _expm1_vec_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean of expm1(X) across rows (axis=0) for a dense 2-D array."""
    n_rows, n_cols = matrix.shape
    result = np.zeros(n_cols)
    for j in nb.prange(n_cols):  # type: ignore[attr-defined]
        s = 0.0
        for i in range(n_rows):
            s += np.expm1(matrix[i, j])
        result[j] = s / n_rows
    return result


def bulk_matrix_arithmetic(matrix: np.ndarray | csr_matrix, is_log1p: bool, axis=0) -> np.ndarray:
    """Arithmetic mean across cells (axis=0), always returned in natural (count) space.

    When is_log1p=True the data is back-transformed via expm1 before taking the
    mean, yielding the true arithmetic mean in count space.  For sparse matrices
    only the stored values are transformed (zeros stay zero, sparsity is preserved).
    When is_log1p=False the arithmetic mean of the raw values is returned directly.
    """
    if is_log1p:
        if isinstance(matrix, csr_matrix):
            m = matrix.copy()
            np.expm1(m.data, out=m.data)
            return np.array(m.mean(axis=axis)).flatten()
        return _expm1_vec_mean(np.asarray(matrix, dtype=np.float64))
    return np.array(matrix.mean(axis=axis)).flatten()


def pseudobulk(
    matrix: np.ndarray | csr_matrix, geometric_mean: bool, is_log1p: bool
) -> np.ndarray:
    """Compute pseudobulk summary across cells (axis=0).

    Always returns values in natural (count) space regardless of ``geometric_mean``
    or ``is_log1p``.

    ``geometric_mean=True`` returns ``expm1(mean(log1p(X)))``, the geometric mean
    back-transformed to count space.  ``is_log1p`` controls whether the log1p step
    is skipped (data already transformed).

    ``geometric_mean=False`` returns the arithmetic mean in count space: when
    ``is_log1p=True`` the data is back-transformed first (``mean(expm1(X))``);
    when ``is_log1p=False`` the arithmetic mean of the raw values is returned.
    """
    if geometric_mean:
        return bulk_matrix_geometric(matrix, is_log1p=is_log1p)
    return bulk_matrix_arithmetic(matrix, is_log1p=is_log1p)


def bulk_matrix_geometric(
    matrix: np.ndarray | csr_matrix, is_log1p: bool, axis=0
) -> np.ndarray:
    """Geometric mean of expression values, back-transformed to count space.

    When is_log1p=True (data already log1p-transformed): expm1(mean(X)).
    When is_log1p=False (raw counts): expm1(mean(log1p(X))).
    Both paths return values in count space.  For sparse matrices only the
    stored values are transformed (log1p(0) = 0, so sparsity is preserved).
    """
    if is_log1p:
        log_mean = np.array(matrix.mean(axis=axis)).flatten()
    elif isinstance(matrix, csr_matrix):
        m = matrix.copy()
        np.log1p(m.data, out=m.data)
        log_mean = np.array(m.mean(axis=axis)).flatten()
    else:
        log_mean = _log1p_col_mean(np.asarray(matrix, dtype=np.float64))
    return _expm1_vec(log_mean)


@nb.njit(parallel=True)
def fold_change(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates the log2-fold change between two arrays."""
    return np.log2(x / y)


@nb.njit(parallel=True)
def percent_change(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates the change between two arrays."""
    return (x - y) / y


def mwu(
    x: np.ndarray | csr_matrix | SparseColumnIndex,
    y: np.ndarray | csr_matrix | SparseColumnIndex,
) -> MannWhitneyUResult:
    """Pass the matrix to the relevant function"""
    if (
        isinstance(x, csr_matrix)
        or isinstance(y, csr_matrix)
        or isinstance(x, SparseColumnIndex)
    ):
        return mannwhitneyu_sparse(x, y)
    else:
        return mannwhitneyu_columns(x, y)
