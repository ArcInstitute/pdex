"""Tests for pdex._utils (set_numba_threadpool)."""

import os

import numba

from pdex._utils import set_numba_threadpool


class TestSetNumbaThreadpool:
    def test_explicit_thread_count(self):
        set_numba_threadpool(4)
        assert numba.get_num_threads() == 4

    def test_zero_uses_all_cpus(self):
        set_numba_threadpool(0)
        expected = os.cpu_count() or 1
        assert numba.get_num_threads() == expected

    def test_single_thread(self):
        set_numba_threadpool(1)
        assert numba.get_num_threads() == 1
