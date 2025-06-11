import anndata as ad
import numpy as np

# A heuristic to determine if the data is log-transformed
# Checks if the mean cell umi count is greater than a certain threshold
# If the the mean cell umi count is < UPPER_LIMIT_LOG, it is assumed that the data is log-transformed
#
# This limit is set to 15 (log-data with >15 average UMI counts would mean an
# average UMI count of ($ e^{15} - 1 = 3.26M $ ) which is unlikely at this point)
UPPER_LIMIT_LOG = 15


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
    sums = adata[mask].X.sum(axis=1)  # type: ignore

    # Determine the mean cell umi count
    mean_umi_count = np.mean(sums)

    # Return True if the mean cell umi count is less than the upper limit
    return bool(mean_umi_count < UPPER_LIMIT_LOG)
