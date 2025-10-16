import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

# A heuristic to determine if the data is log-transformed
# Checks if the mean cell umi count is greater than a certain threshold
# If the the mean cell umi count is < UPPER_LIMIT_LOG, it is assumed that the data is log-transformed
#
# This limit is set to 15 (log-data with >15 average UMI counts would mean an
# average UMI count of ($ e^{15} - 1 = 3.26M $ ) which is unlikely at this point)
UPPER_LIMIT_LOG = 15

EPSILON = 1e-3


def guess_is_log(adata: ad.AnnData, num_cells: int | float = 5e2) -> bool:
    """
    Make an *educated* guess whether the provided anndata is log-transformed.

    Checks whether the any fractional value of the matrix is greater than an epsilon.

    This *cannot* tell the difference between log and normalized data.
    """
    if isinstance(adata.X, csr_matrix) or isinstance(adata.X, csc_matrix):
        frac, _ = np.modf(adata.X.data)
    elif adata.X is None:
        raise ValueError("adata.X is None")
    else:
        frac, _ = np.modf(adata.X)  # type: ignore

    return bool(np.any(frac > EPSILON))
