"""Integration tests for pdex() and _pdex_ref()."""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from pdex2 import DEFAULT_REFERENCE, pdex

EXPECTED_COLUMNS = {
    "target",
    "feature",
    "target_mean",
    "ref_mean",
    "target_membership",
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
        result_groups = set(result["target"].unique().to_list())
        expected_groups = set(small_adata.obs["guide"].unique())
        assert result_groups == expected_groups

    def test_membership_counts(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref")
        for group_name in small_adata.obs["guide"].unique():
            expected_count = (small_adata.obs["guide"] == group_name).sum()
            group_rows = result.filter(pl.col("target") == group_name)
            actual_counts = group_rows["target_membership"].unique().to_list()
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
            group_rows = result.filter(pl.col("target") == group_name)
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

        pdex_group_a = result.filter(pl.col("target") == "A")
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
    """Tests for pdex(..., mode='all') — 1 vs Rest."""

    def test_returns_dataframe(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all")
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all")
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_output_shape(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all")
        n_genes = small_adata.n_vars
        n_groups = len(small_adata.obs["guide"].unique())
        # Every group gets compared against all others
        assert result.shape[0] == n_groups * n_genes

    def test_group_names_present(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all")
        result_groups = set(result["target"].unique().to_list())
        expected_groups = set(small_adata.obs["guide"].unique())
        assert result_groups == expected_groups

    def test_membership_counts(self, small_adata):
        """Each group's membership should match obs, and rest should be total - group."""
        result = pdex(small_adata, groupby="guide", mode="all")
        n_total = small_adata.n_obs
        for group_name in small_adata.obs["guide"].unique():
            expected_group = (small_adata.obs["guide"] == group_name).sum()
            expected_rest = n_total - expected_group
            group_rows = result.filter(pl.col("target") == group_name)
            assert group_rows["target_membership"].unique().to_list() == [expected_group]
            assert group_rows["ref_membership"].unique().to_list() == [expected_rest]

    def test_pvalues_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all")
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_fdr_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all")
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_fold_change_sign(self, small_adata):
        """Group B was boosted the most, so its fold change vs rest should be positive."""
        result = pdex(small_adata, groupby="guide", mode="all")
        group_b_rows = result.filter(pl.col("target") == "B")
        mean_fc = group_b_rows["fold_change"].mean()
        assert mean_fc > 0  # type: ignore

    def test_statistics_against_scipy(self, small_adata):
        """Verify MWU statistics match scipy for group A gene 0 (1 vs rest)."""
        result = pdex(small_adata, groupby="guide", mode="all")

        X = small_adata.X
        obs = small_adata.obs
        group_a_mask = (obs["guide"] == "A").values
        rest_mask = ~group_a_mask

        group_vals = X[group_a_mask, 0]
        rest_vals = X[rest_mask, 0]

        scipy_result = stats.mannwhitneyu(
            group_vals, rest_vals, alternative="two-sided", method="asymptotic"
        )

        pdex_group_a = result.filter(pl.col("target") == "A")
        pdex_pval = pdex_group_a["p_value"][0]
        pdex_stat = pdex_group_a["statistic"][0]

        np.testing.assert_allclose(pdex_stat, scipy_result.statistic, rtol=1e-6)
        np.testing.assert_allclose(pdex_pval, scipy_result.pvalue, rtol=1e-6)

    def test_sparse_returns_dataframe(self, small_adata_sparse):
        result = pdex(small_adata_sparse, groupby="guide", mode="all")
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_sparse_dense_agreement(self, small_adata, small_adata_sparse):
        """Sparse and dense 1vRest results should match."""
        dense_result = pdex(small_adata, groupby="guide", mode="all")
        sparse_result = pdex(small_adata_sparse, groupby="guide", mode="all")

        assert dense_result.shape == sparse_result.shape

        for col in ["p_value", "statistic", "fold_change", "percent_change"]:
            np.testing.assert_allclose(
                dense_result[col].to_numpy(),
                sparse_result[col].to_numpy(),
                rtol=1e-6,
                err_msg=f"Mismatch in column {col}",
            )


class TestPdexOnTargetMode:
    """Tests for pdex(..., mode='on_target')."""

    def test_returns_dataframe(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_output_shape(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        n_groups = on_target_adata.obs["guide"].nunique() - 1  # control excluded
        assert result.shape[0] == n_groups

    def test_gene_column_values(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        gene_map = {"A": "gene_1", "B": "gene_2"}
        for row in result.iter_rows(named=True):
            assert row["feature"] == gene_map[row["target"]]

    def test_membership_counts(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        for group_name in result["target"].to_list():
            expected_count = (on_target_adata.obs["guide"] == group_name).sum()
            row = result.filter(pl.col("target") == group_name)
            assert row["target_membership"][0] == expected_count

    def test_ref_membership_is_constant(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        ref_count = (on_target_adata.obs["guide"] == DEFAULT_REFERENCE).sum()
        assert result["ref_membership"].unique().to_list() == [ref_count]

    def test_pvalues_in_range(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_fdr_in_range(self, on_target_adata):
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_statistics_against_scipy(self, on_target_adata):
        """Verify MWU statistic matches scipy for group A at its target gene (gene_1)."""
        result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )

        X = on_target_adata.X
        obs = on_target_adata.obs
        ntc_mask = (obs["guide"] == DEFAULT_REFERENCE).values
        group_a_mask = (obs["guide"] == "A").values
        gene_idx = list(on_target_adata.var_names).index("gene_1")

        ntc_vals = np.asarray(X[ntc_mask, gene_idx]).ravel()
        group_vals = np.asarray(X[group_a_mask, gene_idx]).ravel()

        scipy_result = stats.mannwhitneyu(
            group_vals, ntc_vals, alternative="two-sided", method="asymptotic"
        )

        row = result.filter(pl.col("target") == "A")
        np.testing.assert_allclose(
            row["statistic"][0], scipy_result.statistic, rtol=1e-6
        )
        np.testing.assert_allclose(row["p_value"][0], scipy_result.pvalue, rtol=1e-6)

    def test_sparse_dense_agreement(self, on_target_adata, on_target_adata_sparse):
        dense_result = pdex(
            on_target_adata, groupby="guide", mode="on_target", gene_col="target_gene"
        )
        sparse_result = pdex(
            on_target_adata_sparse,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
        )

        assert dense_result.shape == sparse_result.shape
        for col in ["p_value", "statistic", "fold_change", "percent_change"]:
            np.testing.assert_allclose(
                dense_result[col].to_numpy(),
                sparse_result[col].to_numpy(),
                rtol=1e-6,
                err_msg=f"Mismatch in column {col}",
            )


class TestPdexOnTargetValidation:
    def test_missing_gene_col_kwarg(self, on_target_adata):
        with pytest.raises(ValueError, match="gene_col"):
            pdex(on_target_adata, groupby="guide", mode="on_target")

    def test_missing_gene_col_column(self, small_adata):
        with pytest.raises(ValueError, match="Missing column"):
            pdex(small_adata, groupby="guide", mode="on_target", gene_col="nonexistent")

    def test_ambiguous_group_gene_mapping(self, on_target_adata):
        """A group with two different target_gene values should raise."""
        adata = on_target_adata.copy()
        # Give group A two different target genes
        a_indices = adata.obs[adata.obs["guide"] == "A"].index
        adata.obs.loc[a_indices[0], "target_gene"] = "gene_3"
        with pytest.raises(ValueError, match="map to multiple genes"):
            pdex(adata, groupby="guide", mode="on_target", gene_col="target_gene")

    def test_unknown_gene_name_warns_and_skips(self, on_target_adata):
        """A target gene not in var_names should warn and skip that group."""
        adata = on_target_adata.copy()
        adata.obs.loc[adata.obs["guide"] == "A", "target_gene"] = "not_a_real_gene"
        with pytest.warns(UserWarning, match="not_a_real_gene"):
            result = pdex(
                adata, groupby="guide", mode="on_target", gene_col="target_gene"
            )
        assert "A" not in result["target"].to_list()
        assert result.shape[0] == adata.obs["guide"].nunique() - 2  # control + A excluded


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
