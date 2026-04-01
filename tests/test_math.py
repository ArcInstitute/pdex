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


class TestFoldChangeWithPriorCount:
    def test_zero_prior_count_matches_baseline(self):
        """prior_count=0.0 must be identical to calling without it."""
        x = np.array([4.0, 8.0, 0.1])
        y = np.array([2.0, 4.0, 0.001])
        np.testing.assert_array_equal(fold_change(x, y), fold_change(x, y, 0.0))

    def test_dampens_extreme_fc_from_near_zero_denominator(self):
        """prior_count=0.5 pulls extreme FC toward zero."""
        x = np.array([0.1])
        y = np.array([0.001])
        fc_raw = fold_change(x, y)[0]
        fc_dampened = fold_change(x, y, 0.5)[0]
        assert abs(fc_dampened) < abs(fc_raw)
        np.testing.assert_allclose(fc_dampened, np.log2(0.6 / 0.501), rtol=1e-5)

    def test_preserves_direction(self):
        """prior_count should not flip the sign of fold change."""
        x = np.array([2.0, 0.5])
        y = np.array([1.0, 1.0])
        result = fold_change(x, y, 0.5)
        assert result[0] > 0
        assert result[1] < 0

    def test_equal_means_still_zero(self):
        """When target_mean == ref_mean, FC should be 0 regardless of prior_count."""
        x = np.array([0.5, 2.0])
        result = fold_change(x, x, 0.5)
        np.testing.assert_allclose(result, [0.0, 0.0])


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
