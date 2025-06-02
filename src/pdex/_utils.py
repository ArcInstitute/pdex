import anndata as ad
import numpy as np


def guess_is_log(adata: ad.AnnData, num_cells: int | float = 5e2) -> bool:
    """Make an *educated* guess whether the provided anndata is log-transformed.

    Selects a random subset of cells and sums their counts.
    Returns false if all decimal components are zero (unlikely for log transformed data)
    """
    # Select either the provided `num_cells` or the maximum number of cells in the `adata`
    num_cells = int(min(num_cells, adata.shape[0]))

    # Draw a random mask of cells
    mask = np.random.choice(adata.shape[0], size=num_cells, replace=False)

    # Sum the matrix across the selected cell subset
    sums = adata[mask].X.sum(axis=1)

    # Extract the fractional components of the array
    decimals, _ = np.modf(sums)

    return np.any(decimals != 0.0)
