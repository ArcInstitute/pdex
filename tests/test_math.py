"""Tests for pdex._math (log2_fold_change, percent_change, bulk_matrix_geometric)."""

import numpy as np
import pytest
from scipy import stats
from scipy.sparse import csr_matrix

from pdex._math import (
    bulk_matrix_arithmetic,
    bulk_matrix_geometric,
    bulk_matrix_pre_transform_mean,
    cpm_bulk,
    cpm_from_gene_means,
    log2_fold_change,
    mwu_one_vs_rest,
    percent_change,
    pseudobulk,
    pseudobulk_from_pre_mean,
)


class TestFoldChange:
    def test_ratio_of_two(self):
        x = np.array([4.0, 8.0])
        y = np.array([2.0, 4.0])
        result = log2_fold_change(x, y)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_equal_values(self):
        x = np.array([3.0, 5.0])
        result = log2_fold_change(x, x)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_half(self):
        x = np.array([1.0])
        y = np.array([2.0])
        result = log2_fold_change(x, y)
        np.testing.assert_allclose(result, [-1.0])

    def test_known_values(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])
        y = np.array([1.0, 1.0, 1.0, 1.0])
        result = log2_fold_change(x, y)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0])

    def test_zero_over_zero_is_zero(self):
        """0/0 (unexpressed in both groups) is defined as 0.0, not NaN."""
        x = np.array([0.0])
        y = np.array([0.0])
        result = log2_fold_change(x, y)
        assert not np.isnan(result).any()
        np.testing.assert_array_equal(result, [0.0])

    def test_zero_over_zero_mixed_with_finite_and_inf(self):
        """0/0 -> 0.0 while normal ratios and one-sided zeros are untouched."""
        x = np.array([0.0, 4.0, 0.0, 4.0])
        y = np.array([0.0, 2.0, 1.0, 0.0])
        result = log2_fold_change(x, y)
        # 0/0 -> 0.0, log2(2) -> 1.0, log2(0) -> -inf, log2(4/0) -> +inf
        np.testing.assert_array_equal(result, [0.0, 1.0, -np.inf, np.inf])


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

    def test_zero_over_zero_is_zero(self):
        """0/0 (unexpressed in both groups) is defined as 0.0, not NaN."""
        x = np.array([0.0])
        y = np.array([0.0])
        result = percent_change(x, y)
        assert not np.isnan(result).any()
        np.testing.assert_array_equal(result, [0.0])

    def test_zero_over_zero_mixed_with_finite_and_inf(self):
        """0/0 -> 0.0 while normal ratios and a zero reference are untouched."""
        x = np.array([0.0, 4.0, 4.0])
        y = np.array([0.0, 2.0, 0.0])
        result = percent_change(x, y)
        # 0/0 -> 0.0, (4-2)/2 -> 1.0, (4-0)/0 -> +inf
        np.testing.assert_array_equal(result, [0.0, 1.0, np.inf])


class TestFoldChangeWithEpsilon:
    def test_zero_epsilon_matches_baseline(self):
        """epsilon=0.0 must be identical to calling without it."""
        x = np.array([4.0, 8.0, 0.1])
        y = np.array([2.0, 4.0, 0.001])
        np.testing.assert_array_equal(
            log2_fold_change(x, y), log2_fold_change(x, y, 0.0)
        )

    def test_dampens_extreme_fc_from_near_zero_denominator(self):
        """epsilon=0.5 pulls extreme FC toward zero."""
        x = np.array([0.1])
        y = np.array([0.001])
        fc_raw = log2_fold_change(x, y)[0]
        fc_dampened = log2_fold_change(x, y, 0.5)[0]
        assert abs(fc_dampened) < abs(fc_raw)
        np.testing.assert_allclose(fc_dampened, np.log2(0.6 / 0.501), rtol=1e-5)

    def test_preserves_direction(self):
        """epsilon should not flip the sign of fold change."""
        x = np.array([2.0, 0.5])
        y = np.array([1.0, 1.0])
        result = log2_fold_change(x, y, 0.5)
        assert result[0] > 0
        assert result[1] < 0

    def test_equal_means_still_zero(self):
        """When target_mean == ref_mean, FC should be 0 regardless of epsilon."""
        x = np.array([0.5, 2.0])
        result = log2_fold_change(x, x, 0.5)
        np.testing.assert_allclose(result, [0.0, 0.0])


class TestPercentChangeWithPriorCount:
    def test_zero_epsilon_matches_baseline(self):
        """epsilon=0.0 must be identical to calling without it."""
        x = np.array([4.0, 8.0, 0.1])
        y = np.array([2.0, 4.0, 0.001])
        np.testing.assert_array_equal(percent_change(x, y), percent_change(x, y, 0.0))

    def test_dampens_extreme_pc_from_near_zero_denominator(self):
        """epsilon=0.5 pulls extreme percent change toward zero."""
        x = np.array([0.1])
        y = np.array([0.001])
        pc_raw = percent_change(x, y)[0]
        pc_dampened = percent_change(x, y, 0.5)[0]
        assert abs(pc_dampened) < abs(pc_raw)
        np.testing.assert_allclose(
            pc_dampened, (0.1 - 0.001) / (0.001 + 0.5), rtol=1e-5
        )

    def test_preserves_direction(self):
        """epsilon should not flip the sign of percent change."""
        x = np.array([2.0, 0.5])
        y = np.array([1.0, 1.0])
        result = percent_change(x, y, 0.5)
        assert result[0] > 0
        assert result[1] < 0

    def test_equal_means_still_zero(self):
        """When target_mean == ref_mean, percent_change should be 0 regardless of epsilon."""
        x = np.array([0.5, 2.0])
        result = percent_change(x, x, 0.5)
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


class TestCpmBulk:
    """Tests for cpm_bulk (pooled CPM view used by the cpm_filter)."""

    def test_known_values_dense(self):
        """cpm[g] = Σcounts_g / Σcounts_all * 1e6."""
        x = np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 1.0]])
        # gene sums: 9, 9; total 18 -> 0.5 each
        result = cpm_bulk(x, is_log1p=False)
        np.testing.assert_allclose(result, [500_000.0, 500_000.0])

    def test_sums_to_one_million(self):
        x = np.array([[2.0, 4.0, 0.0], [6.0, 1.0, 3.0]])
        np.testing.assert_allclose(cpm_bulk(x, is_log1p=False).sum(), 1e6)

    def test_sparse_matches_dense(self):
        x = np.array([[1.0, 3.0, 0.0], [3.0, 5.0, 2.0], [5.0, 1.0, 0.0]])
        dense = cpm_bulk(x, is_log1p=False)
        sparse = cpm_bulk(csr_matrix(x), is_log1p=False)
        np.testing.assert_allclose(dense, sparse)

    def test_log1p_agrees_with_counts(self):
        """is_log1p=True on log1p(counts) matches is_log1p=False on counts."""
        counts = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 2.0]])
        from_counts = cpm_bulk(counts, is_log1p=False)
        from_log = cpm_bulk(np.log1p(counts), is_log1p=True)
        np.testing.assert_allclose(from_counts, from_log, rtol=1e-10)

    def test_all_zero_is_zero_not_nan(self):
        """An all-zero group yields all-zero CPM (denominator guard), never nan/inf."""
        x = np.zeros((3, 4))
        result = cpm_bulk(x, is_log1p=False)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_scale_invariant(self):
        """Uniformly rescaling counts does not change the CPM (ratio cancels)."""
        x = np.array([[1.0, 3.0, 7.0], [3.0, 5.0, 2.0], [5.0, 1.0, 4.0]])
        np.testing.assert_allclose(cpm_bulk(x, False), cpm_bulk(100.0 * x, False))


class TestCpmFromGeneMeans:
    """Tests for cpm_from_gene_means, the normalize-only half of cpm_bulk."""

    def test_matches_cpm_bulk(self):
        x = np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 1.0]])
        gene_means = bulk_matrix_arithmetic(x, is_log1p=False)
        np.testing.assert_allclose(
            cpm_from_gene_means(gene_means), cpm_bulk(x, is_log1p=False)
        )


class TestBulkMatrixPreTransformMean:
    """Tests for bulk_matrix_pre_transform_mean / pseudobulk_from_pre_mean.

    These power the "all" mode's global-sum-minus-group-sum optimization: the
    round trip must exactly reproduce ``pseudobulk()``, and the linear quantity
    must be safely additive across row partitions.
    """

    @pytest.mark.parametrize("geometric_mean", [True, False])
    @pytest.mark.parametrize("is_log1p", [True, False])
    def test_round_trip_matches_pseudobulk(self, geometric_mean, is_log1p):
        rng = np.random.default_rng(0)
        counts = rng.poisson(5, size=(6, 4)).astype(np.float64)
        x = np.log1p(counts) if is_log1p else counts

        pre_mean = bulk_matrix_pre_transform_mean(
            x, geometric_mean=geometric_mean, is_log1p=is_log1p
        )
        result = pseudobulk_from_pre_mean(pre_mean, geometric_mean)
        expected = pseudobulk(x, geometric_mean=geometric_mean, is_log1p=is_log1p)
        np.testing.assert_allclose(result, expected)

    def test_global_minus_group_equals_rest(self):
        """global_sum - group_sum = rest_sum — the core "all" mode algebraic trick."""
        rng = np.random.default_rng(1)
        x = rng.poisson(5, size=(10, 3)).astype(np.float64)
        group_mask = np.zeros(10, dtype=bool)
        group_mask[:4] = True
        n_total, n_group = x.shape[0], int(group_mask.sum())
        n_rest = n_total - n_group

        pre_mean_global = bulk_matrix_pre_transform_mean(
            x, geometric_mean=True, is_log1p=False
        )
        pre_mean_group = bulk_matrix_pre_transform_mean(
            x[group_mask], geometric_mean=True, is_log1p=False
        )
        rest_pre_mean = (pre_mean_global * n_total - pre_mean_group * n_group) / n_rest
        expected = bulk_matrix_pre_transform_mean(
            x[~group_mask], geometric_mean=True, is_log1p=False
        )
        np.testing.assert_allclose(rest_pre_mean, expected, rtol=1e-10)


def _assert_mwu_one_vs_rest_matches_scipy(matrix, dense_reference, codes, n_groups):
    result = mwu_one_vs_rest(matrix, codes, n_groups)
    for g in range(n_groups):
        group_mask = codes == g
        for j in range(dense_reference.shape[1]):
            col = dense_reference[:, j]
            gv = col[group_mask]
            rv = col[~group_mask]
            scipy_result = stats.mannwhitneyu(
                gv, rv, alternative="two-sided", method="asymptotic"
            )
            np.testing.assert_allclose(
                result.statistic[g, j], scipy_result.statistic, rtol=1e-8, atol=1e-8
            )
            np.testing.assert_allclose(
                result.pvalue[g, j], scipy_result.pvalue, rtol=1e-8, atol=1e-8
            )
    return result


class TestMwuOneVsRest:
    """Tests for the one-shot global-rank 1-vs-rest MWU kernel used by mode='all'."""

    def test_dense_matches_scipy_multi_group(self):
        rng = np.random.default_rng(2)
        x = rng.poisson(5, size=(15, 3)).astype(np.float64)
        x[:5] += 4
        x[10:] = np.clip(x[10:] - 2, 0, None)
        codes = np.array([0] * 5 + [1] * 5 + [2] * 5)
        _assert_mwu_one_vs_rest_matches_scipy(x, x, codes, 3)

    def test_dense_with_ties_matches_scipy(self):
        """Ties spanning multiple groups: tie correction is shared across all groups."""
        x = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [1.0]])
        codes = np.array([0, 0, 1, 1, 2, 2])
        _assert_mwu_one_vs_rest_matches_scipy(x, x, codes, 3)

    def test_group_of_size_one_matches_scipy(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        codes = np.array([0, 1, 1, 1, 1])
        _assert_mwu_one_vs_rest_matches_scipy(x, x, codes, 2)

    def test_all_zero_column_gives_pvalue_one(self):
        """A fully-tied (all-zero) gene needs no special-case: s_sq == 0 -> p = 1.0."""
        x = np.zeros((6, 1))
        codes = np.array([0, 0, 1, 1, 2, 2])
        result = mwu_one_vs_rest(x, codes, 3)
        np.testing.assert_allclose(result.pvalue, 1.0)
        assert np.isfinite(result.statistic).all()

    def test_sparse_matches_dense(self):
        rng = np.random.default_rng(3)
        x = rng.poisson(2, size=(12, 4)).astype(np.float64)
        x[x < 1.5] = 0.0
        codes = np.array([0] * 4 + [1] * 3 + [2] * 5)
        dense_result = mwu_one_vs_rest(x, codes, 3)
        sparse_result = mwu_one_vs_rest(csr_matrix(x), codes, 3)
        np.testing.assert_allclose(
            dense_result.statistic, sparse_result.statistic, rtol=1e-8
        )
        np.testing.assert_allclose(
            dense_result.pvalue, sparse_result.pvalue, rtol=1e-8, atol=1e-10
        )

    def test_sparse_matches_scipy(self):
        rng = np.random.default_rng(4)
        x = rng.poisson(2, size=(12, 3)).astype(np.float64)
        x[x < 1.5] = 0.0
        codes = np.array([0] * 4 + [1] * 3 + [2] * 5)
        _assert_mwu_one_vs_rest_matches_scipy(csr_matrix(x), x, codes, 3)
