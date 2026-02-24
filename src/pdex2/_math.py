import numba as nb
import numpy as np
from numba_mwu import (
    MannWhitneyUResult,
    SparseColumnIndex,
    mannwhitneyu_columns,
    mannwhitneyu_sparse,
)
from scipy.sparse import csr_matrix


def bulk_matrix(matrix: np.ndarray | csr_matrix, axis=0) -> np.ndarray:
    """Calculate the pseudobulk of the matrix across the given axis"""
    return np.array(matrix.mean(axis=axis)).flatten()


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
