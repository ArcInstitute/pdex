import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

EPSILON = 1e-3


def guess_is_log(adata: ad.AnnData) -> bool:
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
