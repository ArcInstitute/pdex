"""Tests for pdex2._utils (set_numba_threadpool)."""

import os

import numba

from pdex2._utils import set_numba_threadpool


class TestSetNumbaThreadpool:
    def test_explicit_thread_count(self):
        set_numba_threadpool(4)
        assert numba.config.NUMBA_NUM_THREADS == 4  # type : ignore

    def test_zero_uses_all_cpus(self):
        set_numba_threadpool(0)
        expected = os.cpu_count() or 1
        assert numba.config.NUMBA_NUM_THREADS == expected  # type : ignore

    def test_single_thread(self):
        set_numba_threadpool(1)
        assert numba.config.NUMBA_NUM_THREADS == 1  # type : ignore
