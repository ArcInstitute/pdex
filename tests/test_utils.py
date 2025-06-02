import numpy as np

from pdex._utils import guess_is_log


def test_log_guess():
    log_matrix = np.random.random(size=(1000, 10))
    assert guess_is_log(log_matrix)

    count_matrix = np.random.randint(0, 1e6, size=(1000, 10))
    assert not guess_is_log(count_matrix)
