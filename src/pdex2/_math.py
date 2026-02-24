import numba as nb
import numpy as np


@nb.njit
def fold_change(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates the log2-fold change between two arrays."""
    return np.log2(x / y)


@nb.njit
def percent_change(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculates the change between two arrays."""
    return (x - y) / y
