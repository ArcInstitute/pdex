"""Integration tests for pdex() and _pdex_ref()."""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from pdex2 import DEFAULT_REFERENCE, pdex

EXPECTED_COLUMNS = {
    "group",
    "group_mean",
    "ref_mean",
    "group_membership",
    "ref_membership",
    "fold_change",
    "percent_change",
    "p_value",
    "statistic",
    "fdr",
}


class TestPdexRefMode:
    """Tests for pdex(..., mode='ref')."""

    def test_returns_dataframe(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_output_shape(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        n_genes = small_adata.n_vars
        n_groups = len(small_adata.obs["guide"].unique())
        # All groups (including reference) get a row per gene
        assert result.shape[0] == n_groups * n_genes

    def test_group_names_present(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        result_groups = set(result["group"].unique().to_list())
        expected_groups = set(small_adata.obs["guide"].unique())
        assert result_groups == expected_groups

    def test_membership_counts(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        for group_name in small_adata.obs["guide"].unique():
            expected_count = (small_adata.obs["guide"] == group_name).sum()
            group_rows = result.filter(pl.col("group") == group_name)
            actual_counts = group_rows["group_membership"].unique().to_list()
            assert actual_counts == [expected_count]

    def test_ref_membership_is_constant(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        ref_count = (small_adata.obs["guide"] == DEFAULT_REFERENCE).sum()
        assert result["ref_membership"].unique().to_list() == [ref_count]

    def test_pvalues_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_fdr_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_fold_change_sign(self, small_adata):
        """Groups A and B have higher expression than non-targeting,
        so fold change should be positive."""
        result = pdex(small_adata, groupby="guide", mode="ref")
        for group_name in ["A", "B"]:
            group_rows = result.filter(pl.col("group") == group_name)
            # Mean fold change should be positive since we boosted these groups
            mean_fc = group_rows["fold_change"].mean()
            assert mean_fc > 0, f"Expected positive fold change for group {group_name}"  # type: ignore

    def test_statistics_against_scipy(self, small_adata):
        """Verify MWU statistics match scipy for at least one group/gene pair."""
        result = pdex(small_adata, groupby="guide", mode="ref")

        X = small_adata.X
        obs = small_adata.obs
        ntc_mask = obs["guide"] == DEFAULT_REFERENCE
        group_a_mask = obs["guide"] == "A"

        # Compare gene 0 for group A
        ntc_vals = X[ntc_mask.values, 0]
        group_vals = X[group_a_mask.values, 0]

        scipy_result = stats.mannwhitneyu(
            group_vals, ntc_vals, alternative="two-sided", method="asymptotic"
        )

        pdex_group_a = result.filter(pl.col("group") == "A")
        pdex_pval = pdex_group_a["p_value"][0]
        pdex_stat = pdex_group_a["statistic"][0]

        np.testing.assert_allclose(pdex_stat, scipy_result.statistic, rtol=1e-6)
        np.testing.assert_allclose(pdex_pval, scipy_result.pvalue, rtol=1e-6)

    def test_custom_reference(self, small_adata):
        """Using group A as reference instead of default."""
        result = pdex(small_adata, groupby="guide", mode="ref", reference="A")
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] > 0


class TestPdexRefSparse:
    """Tests for pdex with sparse CSR input."""

    def test_returns_dataframe(self, small_adata_sparse):
        result = pdex(small_adata_sparse, groupby="guide", mode="ref")
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, small_adata_sparse):
        result = pdex(small_adata_sparse, groupby="guide", mode="ref")
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_sparse_dense_agreement(self, small_adata, small_adata_sparse):
        """Sparse and dense inputs should produce the same results."""
        dense_result = pdex(small_adata, groupby="guide", mode="ref")
        sparse_result = pdex(small_adata_sparse, groupby="guide", mode="ref")

        assert dense_result.shape == sparse_result.shape

        for col in ["p_value", "statistic", "fold_change", "percent_change"]:
            np.testing.assert_allclose(
                dense_result[col].to_numpy(),
                sparse_result[col].to_numpy(),
                rtol=1e-6,
                err_msg=f"Mismatch in column {col}",
            )


class TestPdexAllMode:
    def test_returns_empty_dataframe(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all")
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 0


class TestPdexValidation:
    def test_invalid_mode(self, small_adata):
        with pytest.raises(ValueError, match="Invalid mode"):
            pdex(
                small_adata,
                groupby="guide",
                mode="invalid",  # type: ignore
            )

    def test_missing_groupby(self, small_adata):
        with pytest.raises(ValueError, match="Missing column"):
            pdex(small_adata, groupby="nonexistent", mode="ref")

    def test_missing_reference(self, small_adata):
        with pytest.raises(ValueError, match="Missing reference"):
            pdex(small_adata, groupby="guide", mode="ref", reference="does_not_exist")
