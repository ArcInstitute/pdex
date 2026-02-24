import os

import numba


def set_numba_threadpool(threads: int = 0):
    if threads == 0:
        available_threads = os.cpu_count()
        if not available_threads:
            threads = 1
        else:
            threads = available_threads

    numba.config.NUMBA_NUM_THREADS = threads  # type: ignore
