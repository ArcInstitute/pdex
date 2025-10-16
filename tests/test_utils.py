import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from pdex._utils import guess_is_log

N_CELLS = 1000
N_GENES = 10
MAX_COUNT = 1e6


def build_anndata(log=False, sparse: str | None = None) -> ad.AnnData:
    dim = (N_CELLS, N_GENES)
    matrix = np.random.random(size=dim)
    if sparse == "csr":
        matrix = csr_matrix(matrix)
    elif sparse == "csc":
        matrix = csc_matrix(matrix)
    return ad.AnnData(
        X=matrix if log else np.random.randint(0, int(MAX_COUNT), size=dim)
    )


def test_log_guess_logtrue():
    adata = build_anndata(log=True)
    assert guess_is_log(adata)

    adata = build_anndata(log=True, sparse="csc")
    assert guess_is_log(adata)

    adata = build_anndata(log=True, sparse="csr")
    assert guess_is_log(adata)


def test_log_guess_logfalse():
    adata = build_anndata(log=False)
    assert not guess_is_log(adata)

    adata = build_anndata(log=False, sparse="csc")
    assert not guess_is_log(adata)

    adata = build_anndata(log=False, sparse="csr")
    assert not guess_is_log(adata)
