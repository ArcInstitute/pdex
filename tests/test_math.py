"""Tests for pdex2._math (fold_change, percent_change)."""

import numpy as np

from pdex2._math import fold_change, percent_change


class TestFoldChange:
    def test_ratio_of_two(self):
        x = np.array([4.0, 8.0])
        y = np.array([2.0, 4.0])
        result = fold_change(x, y)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_equal_values(self):
        x = np.array([3.0, 5.0])
        result = fold_change(x, x)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_half(self):
        x = np.array([1.0])
        y = np.array([2.0])
        result = fold_change(x, y)
        np.testing.assert_allclose(result, [-1.0])

    def test_known_values(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])
        y = np.array([1.0, 1.0, 1.0, 1.0])
        result = fold_change(x, y)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0])


class TestPercentChange:
    def test_double(self):
        x = np.array([4.0, 10.0])
        y = np.array([2.0, 5.0])
        result = percent_change(x, y)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_equal_values(self):
        x = np.array([3.0, 5.0])
        result = percent_change(x, x)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_decrease(self):
        x = np.array([1.0])
        y = np.array([2.0])
        result = percent_change(x, y)
        np.testing.assert_allclose(result, [-0.5])

    def test_known_values(self):
        x = np.array([5.0, 10.0, 15.0])
        y = np.array([10.0, 10.0, 10.0])
        result = percent_change(x, y)
        np.testing.assert_allclose(result, [-0.5, 0.0, 0.5])
