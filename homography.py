import numpy as np
from common import null
from lmfit import minimize, Parameters


# Container for a homography
class Homography:
    # Initilize estimate for homography
    def __init__(self, srcpts, dstpts):
        assert(srcpts.shape[0] == dstpts.shape[0])
        assert(srcpts.shape[1] == 2)
        assert(dstpts.shape[1] == 2)

        # Store calibration info and estimate homography
        self.srcpts = srcpts
        self.dstpts = dstpts
        self.h = homography_dlt(srcpts, dstpts)

    # Refine guess using numerical optimisation
    def refine(self):
        guess = h_to_params(self.h)
        res = minimize(residual, guess, args=(self.srcpts, self.dstpts))
        self.h = params_to_h(res.params)
        self.h /= self.h[2, 2]


# Convert homography to parameters
def h_to_params(h):
    assert(h.shape == (3, 3))

    params = Parameters()
    for i, hi in zip(range(8), h.reshape(-1)):
        params.add('h' + str(i+1), hi)
    return params


# Convert parameters to homography
def params_to_h(params):
    vals = params.valuesdict()

    h = np.zeros(9)
    for i in range(8):
        h[i] = vals['h' + str(i+1)]
        h[8] += h[i]**2

    return h.reshape(3, 3)


# Project a point using a homography
def transform(m, h):
    assert(m.shape == (2,))
    assert(h.shape == (3, 3))

    p = np.array([m[0], m[1], 1])
    q = h @ p.reshape(-1, 1)
    q = q.reshape(-1)
    q /= q[2]

    return q[:2]


# The residuals of all correspendences
def residual(params, srcpts, dstpts):
    n = len(srcpts)
    h = params_to_h(params)

    res = np.zeros((n, 2))
    for i, src, dst in zip(range(n), srcpts, dstpts):
        clc = transform(src, h)
        res[i] = clc - dst

    return res.reshape(-1)


# Estimate homography by direct linear transform
def homography_dlt(srcpts, dstpts):
    assert(srcpts.shape[0] == dstpts.shape[0])
    assert(srcpts.shape[1] == 2)
    assert(dstpts.shape[1] == 2)

    a = homography_lhs(srcpts, dstpts)
    h = null(a)
    h /= h[-1]

    return h.reshape(3, 3)


# Construct the LHS matrix A
def homography_lhs(srcpts, dstpts):
    assert(srcpts.shape[0] == dstpts.shape[0])
    assert(srcpts.shape[1] == 2)
    assert(dstpts.shape[1] == 2)

    n = len(srcpts)

    lhs = np.zeros((2*n, 9))

    for i, src, dst in zip(range(n), srcpts, dstpts):
        x, y = src
        u, v = dst
    
        # See section 2 of the report
        lhs[i*2, :] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
        lhs[i*2+1, :] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]

    return lhs
