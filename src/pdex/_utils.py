import logging
import os

import numba
import numpy as np
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)


def set_numba_threadpool(threads: int = 0):
    available_threads = os.cpu_count()
    if available_threads is None:
        available_threads = 1

    if threads == 0:
        if not available_threads:
            threads = 1
        else:
            threads = available_threads
    else:
        threads = min(threads, available_threads)

    log.info("Using %d Numba threads (available: %d)", threads, available_threads)
    numba.set_num_threads(threads)


def _detect_is_log1p(X) -> bool:
    """Heuristic: log1p-transformed data has a max value below ~20 (log1p(5e8) ≈ 20)."""
    chunk = X[:500] if X.shape[0] > 500 else X
    if isinstance(chunk, csr_matrix):
        sample = chunk.data  # only stored (non-zero) values
    else:
        sample = np.asarray(chunk).ravel()
    return float(np.max(sample)) < 20.0
