"""Shared fixtures for pdex tests."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_adata(rng):
    """Synthetic AnnData: 3 groups (non-targeting, A, B), 10 cells each, 5 genes."""
    n_cells_per_group = 10
    n_genes = 5
    groups = ["non-targeting", "A", "B"]

    obs_groups = np.repeat(groups, n_cells_per_group)
    n_cells = len(obs_groups)

    # Generate expression: each group has a shifted mean so MWU can detect differences
    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float64)

    # Boost group A and B so they differ from non-targeting
    X[n_cells_per_group : 2 * n_cells_per_group] += 3  # group A
    X[2 * n_cells_per_group :] += 6  # group B

    obs = pd.DataFrame(
        {"guide": obs_groups},
        index=np.array([f"cell_{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(
        index=np.array([f"gene_{i}" for i in range(n_genes)]),
    )

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def small_adata_sparse(small_adata):
    """Same as small_adata but with sparse CSR X matrix."""
    adata = small_adata.copy()
    adata.X = csr_matrix(adata.X)
    return adata


@pytest.fixture
def on_target_adata(small_adata):
    """small_adata with a target_gene obs column mapping each group to one gene."""
    gene_map = {"non-targeting": "gene_0", "A": "gene_1", "B": "gene_2"}
    small_adata.obs["target_gene"] = small_adata.obs["guide"].map(gene_map)
    return small_adata


@pytest.fixture
def on_target_adata_sparse(on_target_adata):
    """on_target_adata with sparse CSR X matrix."""
    adata = on_target_adata.copy()
    adata.X = csr_matrix(adata.X)
    return adata


@pytest.fixture
def cpm_floor_adata(rng):
    """AnnData purpose-built for the cpm_filter: 3 groups, 10 cells each, 5 genes.

    - gene_0..2: well expressed in every group (high CPM, always kept).
    - gene_3: one-sided — zero in the non-targeting reference, expressed in A and B
      (kept by the OR rule; produces a one-sided-zero LFC).
    - gene_4: pure floor — zero in every group (dropped whenever both sides <= T).
    """
    n_per = 10
    groups = ["non-targeting", "A", "B"]
    obs_groups = np.repeat(groups, n_per)
    n_cells = len(obs_groups)
    n_genes = 5

    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float64)
    X[n_per : 2 * n_per, :3] += 3  # boost genes 0..2 in A
    X[2 * n_per :, :3] += 6  # boost genes 0..2 in B

    # gene_3: zero in reference, expressed in A and B (one-sided)
    X[:n_per, 3] = 0.0
    # gene_4: zero everywhere (pure floor)
    X[:, 4] = 0.0

    obs = pd.DataFrame(
        {"guide": obs_groups},
        index=np.array([f"cell_{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(index=np.array([f"gene_{i}" for i in range(n_genes)]))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def cpm_floor_adata_sparse(cpm_floor_adata):
    """cpm_floor_adata with sparse CSR X matrix."""
    adata = cpm_floor_adata.copy()
    adata.X = csr_matrix(adata.X)
    return adata


@pytest.fixture
def small_adata_log1p(small_adata):
    """small_adata with X replaced by log1p-transformed values."""
    adata = small_adata.copy()
    adata.X = np.log1p(adata.X)
    return adata


@pytest.fixture
def small_adata_backed(small_adata, tmp_path):
    """small_adata written to disk and re-opened in backed mode."""
    path = tmp_path / "test.h5ad"
    small_adata.write_h5ad(path)
    return ad.read_h5ad(path, backed="r")
