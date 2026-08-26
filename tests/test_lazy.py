"""Lazy/backed AnnData support.

Every test asserts the invariant that matters: an AnnData opened lazily via
``anndata.experimental.read_lazy`` (h5ad or zarr store, dask X, xarray obs) or
in backed h5ad mode produces the same results as the in-memory equivalent, and
``mode="all"``'s gene-block streaming is invariant to ``block_size``.
"""

import anndata as ad
import pytest
from polars.testing import assert_frame_equal

from pdex import pdex

pytest.importorskip("dask", reason="lazy IO requires the pdex[lazy] extra")
pytest.importorskip("xarray", reason="lazy IO requires the pdex[lazy] extra")

FORMATS = ["h5ad", "zarr"]


def _read_lazy(adata: ad.AnnData, tmp_path, fmt: str) -> ad.AnnData:
    from anndata.experimental import read_lazy

    path = tmp_path / f"data.{fmt}"
    if fmt == "h5ad":
        adata.write_h5ad(path)
    else:
        adata.write_zarr(path)
    return read_lazy(path)


def _run(adata: ad.AnnData, mode: str, **kwargs):
    result = pdex(adata, groupby="guide", mode=mode, is_log1p=False, **kwargs)  # ty: ignore[invalid-argument-type]
    return result.sort(["target", "feature"])


class TestReadLazyMatchesInMemory:
    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("fixture", ["small_adata", "small_adata_sparse"])
    def test_ref_mode(self, request, tmp_path, fmt, fixture):
        adata = request.getfixturevalue(fixture)
        expected = _run(adata, "ref")
        lazy = _read_lazy(adata, tmp_path, fmt)
        assert_frame_equal(_run(lazy, "ref"), expected)

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize(
        "fixture", ["multi_group_adata", "multi_group_adata_sparse"]
    )
    def test_all_mode(self, request, tmp_path, fmt, fixture):
        adata = request.getfixturevalue(fixture)
        expected = _run(adata, "all")
        lazy = _read_lazy(adata, tmp_path, fmt)
        assert_frame_equal(_run(lazy, "all"), expected)

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("fixture", ["on_target_adata", "on_target_adata_sparse"])
    def test_on_target_mode(self, request, tmp_path, fmt, fixture):
        adata = request.getfixturevalue(fixture)
        expected = _run(adata, "on_target", gene_col="target_gene")
        lazy = _read_lazy(adata, tmp_path, fmt)
        assert_frame_equal(_run(lazy, "on_target", gene_col="target_gene"), expected)

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("mode", ["ref", "all"])
    def test_cpm_filter(self, cpm_floor_adata, tmp_path, fmt, mode):
        expected = _run(cpm_floor_adata, mode, cpm_filter=5.0)
        lazy = _read_lazy(cpm_floor_adata, tmp_path, fmt)
        assert_frame_equal(_run(lazy, mode, cpm_filter=5.0), expected)

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_cpm_filter_on_target(self, on_target_adata, tmp_path, fmt):
        """Exercises the streaming per-cell library-size pass on a lazy X."""
        expected = _run(
            on_target_adata, "on_target", gene_col="target_gene", cpm_filter=1.0
        )
        lazy = _read_lazy(on_target_adata, tmp_path, fmt)
        assert_frame_equal(
            _run(lazy, "on_target", gene_col="target_gene", cpm_filter=1.0), expected
        )

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_is_log1p_autodetect(self, small_adata, tmp_path, fmt):
        """_detect_is_log1p must realize dask samples instead of choking on them."""
        lazy = _read_lazy(small_adata, tmp_path, fmt)
        result = pdex(lazy, groupby="guide", mode="ref", is_log1p=None)
        assert result.height > 0


class TestGeneBlockStreaming:
    """mode='all' results must be invariant to block_size."""

    @pytest.mark.parametrize("block_size", [1, 3, 100])
    @pytest.mark.parametrize(
        "fixture", ["multi_group_adata", "multi_group_adata_sparse"]
    )
    def test_in_memory_blocked_identical(self, request, fixture, block_size):
        adata = request.getfixturevalue(fixture)
        expected = _run(adata, "all")
        blocked = _run(adata, "all", block_size=block_size)
        assert_frame_equal(blocked, expected)

    @pytest.mark.parametrize("block_size", [2, 3])
    def test_cpm_filter_blocked_identical(self, cpm_floor_adata, block_size):
        """CPM normalizes across all genes — a per-block CPM would silently
        diverge, so this guards the cross-block reduction ordering."""
        expected = _run(cpm_floor_adata, "all", cpm_filter=5.0)
        blocked = _run(cpm_floor_adata, "all", cpm_filter=5.0, block_size=block_size)
        assert_frame_equal(blocked, expected)

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("block_size", [3])
    def test_lazy_blocked(self, multi_group_adata, tmp_path, fmt, block_size):
        expected = _run(multi_group_adata, "all")
        lazy = _read_lazy(multi_group_adata, tmp_path, fmt)
        assert_frame_equal(_run(lazy, "all", block_size=block_size), expected)


class TestBackedH5ad:
    """Classic ad.read_h5ad(backed='r') keeps working across all modes."""

    @pytest.mark.parametrize("mode", ["ref", "all"])
    def test_modes(self, small_adata, small_adata_backed, mode):
        expected = _run(small_adata, mode)
        assert_frame_equal(_run(small_adata_backed, mode), expected)

    def test_backed_sparse(self, small_adata_sparse, tmp_path):
        path = tmp_path / "sparse.h5ad"
        small_adata_sparse.write_h5ad(path)
        backed = ad.read_h5ad(path, backed="r")
        expected = _run(small_adata_sparse, "all")
        assert_frame_equal(_run(backed, "all"), expected)


class TestRemoteLikeStore:
    """read_lazy over an fsspec store exercises the same code path as s3://."""

    def test_memory_store_zarr(self, small_adata):
        zarr = pytest.importorskip("zarr")
        pytest.importorskip("fsspec")
        from anndata.experimental import read_lazy

        store = zarr.storage.FsspecStore.from_url("memory://pdex_test.zarr")
        small_adata.write_zarr(store)
        lazy = read_lazy(store)

        expected = _run(small_adata, "ref")
        assert_frame_equal(_run(lazy, "ref"), expected)
