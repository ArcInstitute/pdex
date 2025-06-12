import anndata as ad
import numpy as np

from pdex._utils import guess_is_log

N_CELLS = 1000
N_GENES = 10
MAX_COUNT = 1e6


def build_anndata(log=False) -> ad.AnnData:
    dim = (N_CELLS, N_GENES)
    return ad.AnnData(
        X=np.random.random(size=dim)
        if log
        else np.random.randint(0, int(MAX_COUNT), size=dim)
    )


def test_log_guess():
    log_anndata = build_anndata(log=True)
    assert guess_is_log(log_anndata)

    count_anndata = build_anndata(log=False)
    assert not guess_is_log(count_anndata)
