import os

import numba


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

    numba.set_num_threads(threads)
