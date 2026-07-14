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


@pytest.fixture
def multi_group_adata(rng):
    """Synthetic AnnData for stress-testing 1-vs-rest across many uneven groups.

    6 groups (sizes 1, 3, 7, 5, 2, 6 -> 24 cells), 8 genes. Most genes have
    group-specific shifts; gene_6 is heavily tied (two distinct values) and
    gene_7 is all-zero, exercising the degenerate (s_sq <= 0) MWU case.
    """
    group_sizes = {"g0": 1, "g1": 3, "g2": 7, "g3": 5, "g4": 2, "g5": 6}
    offsets = {"g0": 0, "g1": 2, "g2": 0, "g3": 5, "g4": 0, "g5": -2}
    n_genes = 8

    obs_groups = np.concatenate(
        [np.repeat(name, size) for name, size in group_sizes.items()]
    )
    n_cells = len(obs_groups)

    X = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float64)
    idx = 0
    for name, size in group_sizes.items():
        X[idx : idx + size] += offsets[name]
        idx += size
    X = np.clip(X, 0, None)

    # gene_6: heavily tied (only two distinct values across all cells)
    X[:, 6] = (rng.random(n_cells) > 0.5).astype(np.float64) * 3.0
    # gene_7: zero everywhere (degenerate MWU case)
    X[:, 7] = 0.0

    obs = pd.DataFrame(
        {"guide": obs_groups},
        index=np.array([f"cell_{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(index=np.array([f"gene_{i}" for i in range(n_genes)]))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def multi_group_adata_sparse(multi_group_adata):
    """multi_group_adata with sparse CSR X matrix."""
    adata = multi_group_adata.copy()
    adata.X = csr_matrix(adata.X)
    return adata
