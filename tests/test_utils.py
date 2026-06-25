"""Tests for pdex._utils (set_numba_threadpool)."""

import numba

from pdex._utils import _available_cpus, set_numba_threadpool


class TestSetNumbaThreadpool:
    def test_explicit_thread_count(self):
        set_numba_threadpool(4)
        assert numba.get_num_threads() == 4

    def test_zero_uses_all_available_cpus(self):
        set_numba_threadpool(0)
        assert numba.get_num_threads() == _available_cpus()

    def test_single_thread(self):
        set_numba_threadpool(1)
        assert numba.get_num_threads() == 1
