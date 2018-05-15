import numpy as np
from common import null
from lmfit import minimize, Parameters


class Homography:
    def __init__(self, srcpts, dstpts):
        assert (len(srcpts) == len(dstpts))
        self.srcpts = srcpts
        self.dstpts = dstpts
        self.h = homography_dlt(srcpts, dstpts)

    def refine(self):
        guess = h_to_params(self.h)
        res = minimize(residual, guess, args=(self.srcpts, self.dstpts))
        self.h = params_to_h(res.params)
        self.h /= self.h[2, 2]


def h_to_params(h):
    params = Parameters()
    for i, hi in zip(range(8), h.reshape(-1)):
        params.add('h' + str(i+1), hi)
    return params


def params_to_h(params):
    vals = params.valuesdict()
    h = np.zeros(9)
    for i in range(8):
        h[i] = vals['h' + str(i+1)]
        h[8] += h[i]**2
    return h.reshape(3, 3)


def transform(m, h):
    # p = m.append(1)
    p = np.array([m[0], m[1], 1])
    q = h @ p.reshape(-1, 1)
    q = q.reshape(-1)
    q /= q[2]
    return q[:2]


def residual(params, srcpts, dstpts):
    n = len(srcpts)
    h = params_to_h(params)

    res = np.zeros((n, 2))
    for i, src, dst in zip(range(n), srcpts, dstpts):
        clc = transform(src, h)
        res[i] = clc - dst

    return res.reshape(-1)


def homography_dlt(srcpts, dstpts):
    a = homography_lhs(srcpts, dstpts)
    h = null(a)
    h /= h[-1]
    assert (len(h) == 9)
    return h.reshape(3, 3)


def homography_lhs(srcpts, dstpts):
    n = len(srcpts)
    assert(len(srcpts) == len(dstpts))
    lhs = np.zeros((2*n, 9))

    for i, src, dst in zip(range(n), srcpts, dstpts):
        x, y = src
        u, v = dst

        lhs[i*2, 3:9] = [-x, -y, -1, v * x, v * y, v]
        # lhs[i*2, 3] = -x
        # lhs[i*2, 4] = -y
        # lhs[i*2, 5] = -1
        # lhs[i*2, 6] = v * x
        # lhs[i*2, 7] = v * y
        # lhs[i*2, 8] = v

        lhs[i*2+1, :3] = [-x, -y, -1]
        lhs[i*2+1, 6:9] = [u * x, u * y, u]
        # lhs[i*2+1, 0] = -x
        # lhs[i*2+1, 1] = -y
        # lhs[i*2+1, 2] = -1
        # lhs[i*2+1, 6] = u * x
        # lhs[i*2+1, 7] = u * y
        # lhs[i*2+1, 8] = u

    return lhs
