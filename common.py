import numpy as np


def null(mtx):
    _, _, vh = np.linalg.svd(mtx)
    return vh[-1]
