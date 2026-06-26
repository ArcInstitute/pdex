"""Integration tests for pdex() and _pdex_ref()."""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from pdex import DEFAULT_REFERENCE, pdex

EXPECTED_COLUMNS = {
    "target",
    "feature",
    "target_mean",
    "ref_mean",
    "target_membership",
    "ref_membership",
    "fold_change",
    "log2_fold_change",
    "percent_change",
    "p_value",
    "statistic",
    "fdr",
}


class TestPdexRefMode:
    """Tests for pdex(..., mode='ref')."""

    def test_returns_dataframe(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_output_shape(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        n_genes = small_adata.n_vars
        n_groups = len(small_adata.obs["guide"].unique())
        # Reference group is excluded from the output
        assert result.shape[0] == (n_groups - 1) * n_genes

    def test_reference_excluded_from_output(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        assert DEFAULT_REFERENCE not in result["target"].to_list()

    def test_group_names_present(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        result_groups = set(result["target"].unique().to_list())
        expected_groups = set(small_adata.obs["guide"].unique()) - {DEFAULT_REFERENCE}
        assert result_groups == expected_groups

    def test_membership_counts(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        non_ref_groups = [
            g for g in small_adata.obs["guide"].unique() if g != DEFAULT_REFERENCE
        ]
        for group_name in non_ref_groups:
            expected_count = (small_adata.obs["guide"] == group_name).sum()
            group_rows = result.filter(pl.col("target") == group_name)
            actual_counts = group_rows["target_membership"].unique().to_list()
            assert actual_counts == [expected_count]

    def test_ref_membership_is_constant(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        ref_count = (small_adata.obs["guide"] == DEFAULT_REFERENCE).sum()
        assert result["ref_membership"].unique().to_list() == [ref_count]

    def test_pvalues_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_fdr_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_fold_change_sign(self, small_adata):
        """Groups A and B have higher expression than non-targeting,
        so fold change should be positive."""
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        for group_name in ["A", "B"]:
            group_rows = result.filter(pl.col("target") == group_name)
            # Mean fold change should be positive since we boosted these groups
            mean_fc = group_rows["fold_change"].mean()
            assert mean_fc > 0, f"Expected positive fold change for group {group_name}"  # type: ignore

    def test_statistics_against_scipy(self, small_adata):
        """Verify MWU statistics match scipy for at least one group/gene pair."""
        result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)

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
        result = pdex(
            small_adata, groupby="guide", mode="ref", is_log1p=False, reference="A"
        )
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] > 0

    def test_as_pandas(self, small_adata):
        import pandas as pd

        result = pdex(
            small_adata, groupby="guide", mode="ref", is_log1p=False, as_pandas=True
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_unexpected_kwargs_warns(self, small_adata):
        with pytest.warns(UserWarning, match="typo_arg"):
            pdex(
                small_adata,
                groupby="guide",
                mode="ref",
                is_log1p=False,
                typo_arg="oops",
            )

    def test_epsilon_accepted(self, small_adata):
        """epsilon parameter is accepted without error."""
        result = pdex(small_adata, groupby="guide", is_log1p=False, epsilon=0.5)
        assert isinstance(result, pl.DataFrame)

    def test_default_epsilon_is_tiny_finite_guard(self, small_adata):
        """Omitting epsilon uses the 1e-9 default (not 0.0)."""
        default_result = pdex(small_adata, groupby="guide", is_log1p=False)
        explicit_result = pdex(
            small_adata, groupby="guide", is_log1p=False, epsilon=1e-9
        )
        assert isinstance(default_result, pl.DataFrame)
        assert isinstance(explicit_result, pl.DataFrame)
        assert default_result.equals(explicit_result)


class TestPdexRefSparse:
    """Tests for pdex with sparse CSR input."""

    def test_returns_dataframe(self, small_adata_sparse):
        result = pdex(small_adata_sparse, groupby="guide", mode="ref", is_log1p=False)
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, small_adata_sparse):
        result = pdex(small_adata_sparse, groupby="guide", mode="ref", is_log1p=False)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_sparse_dense_agreement(self, small_adata, small_adata_sparse):
        """Sparse and dense inputs should produce the same results."""
        dense_result = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        sparse_result = pdex(
            small_adata_sparse, groupby="guide", mode="ref", is_log1p=False
        )

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
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_output_shape(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        n_genes = small_adata.n_vars
        n_groups = len(small_adata.obs["guide"].unique())
        # Every group gets compared against all others
        assert result.shape[0] == n_groups * n_genes

    def test_group_names_present(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        result_groups = set(result["target"].unique().to_list())
        expected_groups = set(small_adata.obs["guide"].unique())
        assert result_groups == expected_groups

    def test_membership_counts(self, small_adata):
        """Each group's membership should match obs, and rest should be total - group."""
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        n_total = small_adata.n_obs
        for group_name in small_adata.obs["guide"].unique():
            expected_group = (small_adata.obs["guide"] == group_name).sum()
            expected_rest = n_total - expected_group
            group_rows = result.filter(pl.col("target") == group_name)
            assert group_rows["target_membership"].unique().to_list() == [
                expected_group
            ]
            assert group_rows["ref_membership"].unique().to_list() == [expected_rest]

    def test_pvalues_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_fdr_in_range(self, small_adata):
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_fold_change_sign(self, small_adata):
        """Group B was boosted the most, so its fold change vs rest should be positive."""
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        group_b_rows = result.filter(pl.col("target") == "B")
        mean_fc = group_b_rows["fold_change"].mean()
        assert mean_fc > 0  # type: ignore

    def test_statistics_against_scipy(self, small_adata):
        """Verify MWU statistics match scipy for group A gene 0 (1 vs rest)."""
        result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)

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
        result = pdex(small_adata_sparse, groupby="guide", mode="all", is_log1p=False)
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_sparse_dense_agreement(self, small_adata, small_adata_sparse):
        """Sparse and dense 1vRest results should match."""
        dense_result = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        sparse_result = pdex(
            small_adata_sparse, groupby="guide", mode="all", is_log1p=False
        )

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
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(self, on_target_adata):
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_output_shape(self, on_target_adata):
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        n_groups = on_target_adata.obs["guide"].nunique() - 1  # control excluded
        assert result.shape[0] == n_groups

    def test_gene_column_values(self, on_target_adata):
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        gene_map = {"A": "gene_1", "B": "gene_2"}
        for row in result.iter_rows(named=True):
            assert row["feature"] == gene_map[row["target"]]

    def test_membership_counts(self, on_target_adata):
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        for group_name in result["target"].to_list():
            expected_count = (on_target_adata.obs["guide"] == group_name).sum()
            row = result.filter(pl.col("target") == group_name)
            assert row["target_membership"][0] == expected_count

    def test_ref_membership_is_constant(self, on_target_adata):
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        ref_count = (on_target_adata.obs["guide"] == DEFAULT_REFERENCE).sum()
        assert result["ref_membership"].unique().to_list() == [ref_count]

    def test_pvalues_in_range(self, on_target_adata):
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_fdr_in_range(self, on_target_adata):
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        assert (result["fdr"] >= 0).all()
        assert (result["fdr"] <= 1).all()

    def test_statistics_against_scipy(self, on_target_adata):
        """Verify MWU statistic matches scipy for group A at its target gene (gene_1)."""
        result = pdex(
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
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
            on_target_adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
        )
        sparse_result = pdex(
            on_target_adata_sparse,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
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
            pdex(on_target_adata, groupby="guide", mode="on_target", is_log1p=False)

    def test_missing_gene_col_column(self, small_adata):
        with pytest.raises(ValueError, match="Missing column"):
            pdex(
                small_adata,
                groupby="guide",
                mode="on_target",
                gene_col="nonexistent",
                is_log1p=False,
            )

    def test_ambiguous_group_gene_mapping(self, on_target_adata):
        """A group with two different target_gene values should raise."""
        adata = on_target_adata.copy()
        # Give group A two different target genes
        a_indices = adata.obs[adata.obs["guide"] == "A"].index
        adata.obs.loc[a_indices[0], "target_gene"] = "gene_3"
        with pytest.raises(ValueError, match="map to multiple genes"):
            pdex(
                adata,
                groupby="guide",
                mode="on_target",
                gene_col="target_gene",
                is_log1p=False,
            )

    def test_unknown_gene_name_warns_and_skips(self, on_target_adata):
        """A target gene not in var_names should warn and skip that group."""
        adata = on_target_adata.copy()
        adata.obs.loc[adata.obs["guide"] == "A", "target_gene"] = "not_a_real_gene"
        with pytest.warns(UserWarning, match="not_a_real_gene"):
            result = pdex(
                adata,
                groupby="guide",
                mode="on_target",
                gene_col="target_gene",
                is_log1p=False,
            )
        assert "A" not in result["target"].to_list()
        assert (
            result.shape[0] == adata.obs["guide"].nunique() - 2
        )  # control + A excluded


class TestPdexValidation:
    def test_negative_epsilon_raises(self, small_adata):
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            pdex(small_adata, groupby="guide", is_log1p=False, epsilon=-0.1)

    def test_invalid_mode(self, small_adata):
        with pytest.raises(ValueError, match="Invalid mode"):
            pdex(
                small_adata,
                groupby="guide",
                mode="invalid",  # type: ignore
                is_log1p=False,
            )

    def test_missing_groupby(self, small_adata):
        with pytest.raises(ValueError, match="Missing column"):
            pdex(small_adata, groupby="nonexistent", mode="ref", is_log1p=False)

    def test_missing_reference(self, small_adata):
        with pytest.raises(ValueError, match="Missing reference"):
            pdex(
                small_adata,
                groupby="guide",
                mode="ref",
                is_log1p=False,
                reference="does_not_exist",
            )


class TestPdexGeometricMean:
    """Tests for is_log1p / geometric_mean behaviour."""

    def test_autodetect_warns(self, small_adata, caplog):
        """Omitting is_log1p should emit a log warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="pdex"):
            pdex(small_adata, groupby="guide", mode="ref", geometric_mean=False)
        assert any("is_log1p not specified" in r.message for r in caplog.records)

    def test_arithmetic_mean_matches_original(self, small_adata):
        """geometric_mean=False, is_log1p=False: target_mean = plain arithmetic mean of raw counts."""
        result = pdex(
            small_adata,
            groupby="guide",
            mode="ref",
            is_log1p=False,
            geometric_mean=False,
        )
        X = small_adata.X
        obs = small_adata.obs
        group_mask = (obs["guide"] == "A").values
        gene_idx = 0
        expected_mean = float(X[group_mask, gene_idx].mean())
        row = result.filter((pl.col("target") == "A") & (pl.col("feature") == "gene_0"))
        np.testing.assert_allclose(row["target_mean"][0], expected_mean, rtol=1e-10)

    def test_arithmetic_mean_is_log1p_true_returns_natural_space(
        self, small_adata_log1p
    ):
        """geometric_mean=False, is_log1p=True: target_mean = mean(expm1(log1p_data)),
        i.e. back-transform first then average."""
        result = pdex(
            small_adata_log1p,
            groupby="guide",
            mode="ref",
            is_log1p=True,
            geometric_mean=False,
        )
        X = small_adata_log1p.X
        obs = small_adata_log1p.obs
        group_mask = (obs["guide"] == "A").values
        gene_idx = 0
        expected_mean = float(np.expm1(X[group_mask, gene_idx]).mean())
        row = result.filter((pl.col("target") == "A") & (pl.col("feature") == "gene_0"))
        np.testing.assert_allclose(row["target_mean"][0], expected_mean, rtol=1e-10)

    def test_geometric_mean_is_log1p_true(self, small_adata_log1p):
        """geometric_mean=True, is_log1p=True: target_mean = expm1(mean(log1p_data))."""
        result = pdex(
            small_adata_log1p,
            groupby="guide",
            mode="ref",
            is_log1p=True,
            geometric_mean=True,
        )
        X = small_adata_log1p.X
        obs = small_adata_log1p.obs
        group_mask = (obs["guide"] == "A").values
        gene_idx = 0
        expected_mean = float(np.expm1(X[group_mask, gene_idx].mean()))
        row = result.filter((pl.col("target") == "A") & (pl.col("feature") == "gene_0"))
        np.testing.assert_allclose(row["target_mean"][0], expected_mean, rtol=1e-10)

    def test_geometric_mean_is_log1p_false(self, small_adata):
        """geometric_mean=True, is_log1p=False: target_mean = expm1(mean(log1p(counts)))."""
        result = pdex(
            small_adata,
            groupby="guide",
            mode="ref",
            is_log1p=False,
            geometric_mean=True,
        )
        X = small_adata.X
        obs = small_adata.obs
        group_mask = (obs["guide"] == "A").values
        gene_idx = 0
        expected_mean = float(np.expm1(np.log1p(X[group_mask, gene_idx]).mean()))
        row = result.filter((pl.col("target") == "A") & (pl.col("feature") == "gene_0"))
        np.testing.assert_allclose(row["target_mean"][0], expected_mean, rtol=1e-10)

    def test_both_log1p_paths_agree(self, small_adata, small_adata_log1p):
        """pdex on raw counts with is_log1p=False and on log1p counts with is_log1p=True
        should yield identical results across all output columns.

        Pseudobulk means back-transform to the same count space, so fold_change and
        percent_change must match. The MWU statistic and p_value operate on the raw
        cell-level values (which differ between the two inputs), so they are NOT
        expected to match — only the pseudobulk-derived columns are tested here.
        """
        raw_result = pdex(
            small_adata,
            groupby="guide",
            mode="ref",
            is_log1p=False,
            geometric_mean=True,
        )
        log_result = pdex(
            small_adata_log1p,
            groupby="guide",
            mode="ref",
            is_log1p=True,
            geometric_mean=True,
        )
        for col in ["target_mean", "ref_mean", "fold_change", "percent_change"]:
            np.testing.assert_allclose(
                raw_result[col].to_numpy(),
                log_result[col].to_numpy(),
                rtol=1e-10,
                err_msg=f"Mismatch in column {col}",
            )

    def test_all_mode_geometric_mean(self, small_adata):
        """geometric_mean=True works in mode='all'."""
        result = pdex(
            small_adata,
            groupby="guide",
            mode="all",
            is_log1p=False,
            geometric_mean=True,
        )
        assert isinstance(result, pl.DataFrame)
        X = small_adata.X
        obs = small_adata.obs
        group_mask = (obs["guide"] == "B").values
        gene_idx = 1
        expected_mean = float(np.expm1(np.log1p(X[group_mask, gene_idx]).mean()))
        row = result.filter((pl.col("target") == "B") & (pl.col("feature") == "gene_1"))
        np.testing.assert_allclose(row["target_mean"][0], expected_mean, rtol=1e-10)


class TestPdexBacked:
    """Backed AnnData should produce the same results as in-memory."""

    def test_ref_mode_backed_matches_inmemory(self, small_adata, small_adata_backed):
        inmem = pdex(small_adata, groupby="guide", mode="ref", is_log1p=False)
        backed = pdex(small_adata_backed, groupby="guide", mode="ref", is_log1p=False)
        assert inmem.shape == backed.shape
        for col in ["p_value", "statistic", "fold_change", "percent_change"]:
            np.testing.assert_allclose(
                inmem[col].to_numpy(),
                backed[col].to_numpy(),
                rtol=1e-6,
                err_msg=f"Mismatch in column {col}",
            )

    def test_all_mode_backed_matches_inmemory(self, small_adata, small_adata_backed):
        inmem = pdex(small_adata, groupby="guide", mode="all", is_log1p=False)
        backed = pdex(small_adata_backed, groupby="guide", mode="all", is_log1p=False)
        assert inmem.shape == backed.shape
        for col in ["p_value", "statistic", "fold_change", "percent_change"]:
            np.testing.assert_allclose(
                inmem[col].to_numpy(),
                backed[col].to_numpy(),
                rtol=1e-6,
                err_msg=f"Mismatch in column {col}",
            )


class TestLog2FoldChangeColumn:
    """Regression test for the `log2_fold_change` column semantics."""

    @pytest.mark.parametrize("mode", ["ref", "all"])
    def test_log2_fold_change_equals_log2_ratio(self, small_adata, mode):
        """log2_fold_change == log2(target_mean / ref_mean) on finite entries."""
        result = pdex(small_adata, groupby="guide", mode=mode, is_log1p=False)
        target = result["target_mean"].to_numpy()
        ref = result["ref_mean"].to_numpy()
        actual = result["log2_fold_change"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.log2(target / ref)
        finite = np.isfinite(expected) & np.isfinite(actual)
        assert finite.any()
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-6)


class TestUnexpressedInBothGroups:
    """A feature unexpressed in both groups (0/0) reports 0.0, not NaN."""

    @pytest.mark.parametrize("mode", ["ref", "all"])
    def test_zero_in_both_is_zero_not_nan(self, small_adata, mode):
        """gene_0 is zero everywhere -> 0/0 in every comparison -> 0.0."""
        adata = small_adata.copy()
        adata.X[:, 0] = 0.0  # gene_0 unexpressed in every cell

        result = pdex(adata, groupby="guide", mode=mode, is_log1p=False, epsilon=0.0)
        gene0 = result.filter(pl.col("feature") == "gene_0")

        assert (gene0["target_mean"].to_numpy() == 0).all()
        assert (gene0["ref_mean"].to_numpy() == 0).all()
        for col in ["log2_fold_change", "fold_change", "percent_change"]:
            values = gene0[col].to_numpy()
            assert not np.isnan(values).any(), f"{col} contains NaN"
            np.testing.assert_array_equal(values, 0.0)

    def test_on_target_zero_in_both_is_zero_not_nan(self, on_target_adata):
        """on_target mode: a targeted gene that is zero everywhere reports 0.0."""
        adata = on_target_adata.copy()
        adata.X[:, 1] = 0.0  # group "A" targets gene_1

        result = pdex(
            adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
            epsilon=0.0,
        )
        row = result.filter(pl.col("target") == "A")
        assert row["target_mean"].to_numpy()[0] == 0
        assert row["ref_mean"].to_numpy()[0] == 0
        for col in ["log2_fold_change", "fold_change", "percent_change"]:
            value = row[col].to_numpy()[0]
            assert not np.isnan(value), f"{col} is NaN"
            assert value == 0.0

    def test_one_sided_zero_still_infinite(self, small_adata):
        """Only 0/0 is filled; a zero target over a nonzero reference stays infinite."""
        adata = small_adata.copy()
        # gene_0 expressed only in the reference -> target_mean 0, ref_mean > 0
        adata.X[:, 0] = 0.0
        adata.X[adata.obs["guide"].to_numpy() == "non-targeting", 0] = 5.0

        result = pdex(adata, groupby="guide", mode="ref", is_log1p=False, epsilon=0.0)
        gene0 = result.filter(pl.col("feature") == "gene_0")
        # log2(0 / ref) -> -inf; percent_change (0 - ref) / ref -> -1.0
        assert np.isneginf(gene0["log2_fold_change"].to_numpy()).all()
        np.testing.assert_allclose(gene0["percent_change"].to_numpy(), -1.0)


def _pairs(df) -> set:
    """Set of (target, feature) tuples in a result frame."""
    return set(zip(df["target"].to_list(), df["feature"].to_list()))


class TestCpmFilter:
    """Tests for the cpm_filter (bulk-CPM floor) parameter."""

    def test_both_sides_below_threshold_dropped(self, cpm_floor_adata):
        """gene_4 (zero in every group) is dropped from the output."""
        result = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        assert "gene_4" not in result["feature"].to_list()

    def test_one_side_above_threshold_kept(self, cpm_floor_adata):
        """gene_3 (zero in ref, expressed in target) survives via the OR rule."""
        result = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        assert "gene_3" in result["feature"].to_list()

    def test_negative_threshold_keeps_everything(self, cpm_floor_adata):
        """A negative threshold keeps all genes (CPM >= 0 > T)."""
        unfiltered = pdex(cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False)
        result = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=-1
        )
        assert _pairs(result) == _pairs(unfiltered)

    def test_zero_threshold_strict_drops_only_exact_zero(self, cpm_floor_adata):
        """T=0 drops genes whose pooled CPM is exactly 0 (strict >), keeps the rest."""
        result = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=0.0
        )
        features = set(result["feature"].to_list())
        assert "gene_4" not in features  # cpm 0, not > 0 -> dropped
        assert {"gene_0", "gene_1", "gene_2", "gene_3"} <= features

    def test_none_matches_unfiltered(self, cpm_floor_adata):
        """cpm_filter=None is identical to omitting it."""
        omitted = pdex(cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False)
        explicit = pdex(
            cpm_floor_adata,
            groupby="guide",
            mode="ref",
            is_log1p=False,
            cpm_filter=None,
        )
        assert isinstance(omitted, pl.DataFrame)
        assert isinstance(explicit, pl.DataFrame)
        assert omitted.equals(explicit)

    def test_filtered_is_subset_with_unchanged_values(self, cpm_floor_adata):
        """Surviving rows keep their exact means/p-values; only the row set shrinks."""
        unfiltered = pdex(cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False)
        filtered = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        assert _pairs(filtered) < _pairs(unfiltered)  # strict subset
        # Surviving rows keep their exact target_mean / p_value (look up by key)
        full = {
            (t, f): (tm, pv)
            for t, f, tm, pv in zip(
                unfiltered["target"].to_list(),
                unfiltered["feature"].to_list(),
                unfiltered["target_mean"].to_list(),
                unfiltered["p_value"].to_list(),
            )
        }
        for t, f, tm, pv in zip(
            filtered["target"].to_list(),
            filtered["feature"].to_list(),
            filtered["target_mean"].to_list(),
            filtered["p_value"].to_list(),
        ):
            np.testing.assert_allclose(tm, full[(t, f)][0])
            np.testing.assert_allclose(pv, full[(t, f)][1])

    def test_scale_invariant_kept_set(self, cpm_floor_adata):
        """Uniformly rescaling counts does not change which rows survive."""
        scaled = cpm_floor_adata.copy()
        scaled.X = scaled.X * 100.0
        base = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        rescaled = pdex(
            scaled, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        assert _pairs(base) == _pairs(rescaled)

    def test_no_inf_with_default_epsilon(self, cpm_floor_adata):
        """Filter + default epsilon (1e-9) leaves no inf/nan in the LFC columns."""
        result = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        for col in ("log2_fold_change", "percent_change"):
            vals = result[col].to_numpy()
            assert not np.isinf(vals).any(), col
            assert not np.isnan(vals).any(), col

    def test_epsilon_zero_still_allows_one_sided_inf(self, cpm_floor_adata):
        """With epsilon=0, a surviving one-sided zero (gene_3) keeps its +inf LFC."""
        result = pdex(
            cpm_floor_adata,
            groupby="guide",
            mode="ref",
            is_log1p=False,
            cpm_filter=5,
            epsilon=0.0,
        )
        gene3 = result.filter(pl.col("feature") == "gene_3")
        # gene_3: ref_mean 0, target_mean > 0 -> log2(t/0) = +inf
        assert np.isposinf(gene3["log2_fold_change"].to_numpy()).all()

    def test_fdr_over_survivors(self, cpm_floor_adata):
        """FDR is recomputed over surviving genes, not the full gene set."""
        filtered = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        a = filtered.filter(pl.col("target") == "A")
        # FDR matches BH over exactly the surviving p-values
        recomputed = stats.false_discovery_control(a["p_value"].to_numpy())
        np.testing.assert_allclose(a["fdr"].to_numpy(), recomputed)

        # ... and differs from BH over the full (unfiltered) p-value set
        unfiltered = pdex(cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False)
        a_full = unfiltered.filter(pl.col("target") == "A")
        full_fdr = dict(
            zip(
                a_full["feature"].to_list(),
                stats.false_discovery_control(a_full["p_value"].to_numpy()),
            )
        )
        surviving_full = np.array([full_fdr[f] for f in a["feature"].to_list()])
        assert not np.allclose(a["fdr"].to_numpy(), surviving_full)

    def test_sparse_matches_dense(self, cpm_floor_adata, cpm_floor_adata_sparse):
        """Sparse input yields the same kept set and values as dense."""
        dense = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        sparse = pdex(
            cpm_floor_adata_sparse,
            groupby="guide",
            mode="ref",
            is_log1p=False,
            cpm_filter=5,
        )
        assert _pairs(dense) == _pairs(sparse)
        d = dense.sort(["target", "feature"])
        s = sparse.sort(["target", "feature"])
        np.testing.assert_allclose(
            d["target_mean"].to_numpy(), s["target_mean"].to_numpy()
        )

    def test_log1p_kept_set_matches_raw(self, cpm_floor_adata):
        """CPM is computed on counts, so log1p input gives the same kept set."""
        log_adata = cpm_floor_adata.copy()
        log_adata.X = np.log1p(log_adata.X)
        raw = pdex(
            cpm_floor_adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5
        )
        logged = pdex(
            log_adata, groupby="guide", mode="ref", is_log1p=True, cpm_filter=5
        )
        assert _pairs(raw) == _pairs(logged)

    def test_all_mode_drops_floor(self, cpm_floor_adata):
        """In 1-vs-rest mode, the floor gene is dropped for every group."""
        result = pdex(
            cpm_floor_adata, groupby="guide", mode="all", is_log1p=False, cpm_filter=5
        )
        assert "gene_4" not in result["feature"].to_list()
        # every group still present (they retain expressed genes)
        assert set(result["target"].to_list()) == {"non-targeting", "A", "B"}

    def test_on_target_drops_floor_group(self, cpm_floor_adata):
        """on_target: a group whose target gene is a floor gene is dropped."""
        adata = cpm_floor_adata.copy()
        adata.obs["target_gene"] = (
            adata.obs["guide"].map(
                {"non-targeting": "gene_0", "A": "gene_3", "B": "gene_4"}
            )
        ).astype(object)
        result = pdex(
            adata,
            groupby="guide",
            mode="on_target",
            gene_col="target_gene",
            is_log1p=False,
            cpm_filter=5,
        )
        # A targets gene_3 (expressed) -> kept; B targets gene_4 (floor) -> dropped
        assert result["target"].to_list() == ["A"]
        assert result["feature"].to_list() == ["gene_3"]
        # FDR over the single surviving row
        assert (result["fdr"] >= 0).all() and (result["fdr"] <= 1).all()

    def test_all_dropped_returns_empty_with_schema(self, cpm_floor_adata):
        """A threshold above every gene's CPM yields a height-0 frame, full schema."""
        result = pdex(
            cpm_floor_adata,
            groupby="guide",
            mode="ref",
            is_log1p=False,
            cpm_filter=1e12,
        )
        assert result.height == 0
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_negative_values_warn(self, cpm_floor_adata):
        """cpm_filter on data with negative values emits a UserWarning."""
        adata = cpm_floor_adata.copy()
        adata.X[0, 0] = -1.0
        with pytest.warns(UserWarning, match="negative values"):
            pdex(adata, groupby="guide", mode="ref", is_log1p=False, cpm_filter=5)
