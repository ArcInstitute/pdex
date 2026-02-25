import logging
import os

import numba

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
