import logging
import multiprocessing as mp
import os

import numba
import numpy as np
from scipy.sparse import issparse

log = logging.getLogger(__name__)


def _available_cpus() -> int:
    """Return the number of CPUs the current process is allowed to use.

    Uses ``os.sched_getaffinity`` on Linux so SLURM/cgroup/taskset limits are
    respected; falls back to ``multiprocessing.cpu_count`` on macOS/Windows where
    that API is unavailable (those platforms typically run locally without cgroup
    caps). This mirrors how Numba derives ``NUMBA_NUM_THREADS``, so the value never
    exceeds Numba's cap — unlike ``os.cpu_count()``, which reports every physical
    CPU even when affinity or a cgroup restricts the process to far fewer, causing
    ``numba.set_num_threads`` to raise.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


def set_numba_threadpool(threads: int = 0):
    available_threads = _available_cpus()

    if threads == 0:
        threads = available_threads
    else:
        threads = min(threads, available_threads)

    log.info("Using %d Numba threads (available: %d)", threads, available_threads)
    numba.set_num_threads(threads)


def _detect_is_log1p(X) -> bool:
    """Heuristic: log1p-transformed data has a max value below ~20 (log1p(5e8) ≈ 20)."""
    chunk = X[:500] if X.shape[0] > 500 else X
    if issparse(chunk):
        sample = chunk.data  # only stored (non-zero) values
    else:
        sample = np.asarray(chunk).ravel()
    return float(np.max(sample)) < 20.0
