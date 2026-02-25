"""Tests for pdex._math (fold_change, percent_change, bulk_matrix_geometric)."""

import numpy as np

from pdex._math import bulk_matrix_geometric, fold_change, percent_change


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


class TestBulkMatrixGeometric:
    """Tests for bulk_matrix_geometric."""

    def test_is_log1p_false(self):
        """Raw counts: result = expm1(mean(log1p(x))) per gene."""
        # 3 cells x 2 genes
        x = np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 1.0]])
        result = bulk_matrix_geometric(x, is_log1p=False)
        expected = np.expm1(np.log1p(x).mean(axis=0))
        np.testing.assert_allclose(result, expected)

    def test_is_log1p_true(self):
        """Log1p data: result = expm1(mean(X)) per gene."""
        counts = np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 1.0]])
        x = np.log1p(counts)
        result = bulk_matrix_geometric(x, is_log1p=True)
        expected = np.expm1(x.mean(axis=0))
        np.testing.assert_allclose(result, expected)

    def test_both_paths_agree_on_log1p_input(self):
        """Passing raw counts with is_log1p=False and log1p counts with is_log1p=True
        should yield the same geometric mean."""
        counts = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 2.0]])
        raw_result = bulk_matrix_geometric(counts, is_log1p=False)
        log_result = bulk_matrix_geometric(np.log1p(counts), is_log1p=True)
        np.testing.assert_allclose(raw_result, log_result, rtol=1e-10)

    def test_zeros_handled(self):
        """Zeros in raw counts should not cause errors (log1p(0) = 0)."""
        x = np.array([[0.0, 5.0], [0.0, 3.0]])
        result = bulk_matrix_geometric(x, is_log1p=False)
        expected = np.expm1(np.log1p(x).mean(axis=0))
        np.testing.assert_allclose(result, expected)
