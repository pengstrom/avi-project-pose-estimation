import numpy as np
from common import null


class Homography:
    def __init__(self, srcpts, dstpts):
        assert (len(srcpts) == len(dstpts))
        self.srcpts = srcpts
        self.dstpts = dstpts
        self.h = homography_dlt(srcpts, dstpts)

    def refine(self):
        pass


def transform(m, h):
    p = m.append(1)
    q = h @ p.reshape(-1, 1)
    q = q.reshape(-1)
    q /= q[2]
    return q[:2]


def euclid_error(params, x, data):
    print(x)
    h9 = 0
    for a in params:
        h9 += a*a
    h = params.append(h9).reshape(3, 3)

    model = transform(x, h)
    return np.float64([data[0]-model[0], data[1]-model[1]])


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

    for i in range(0, n):
        x, y = srcpts[i]
        u, v = dstpts[i]

        lhs[i*2, 3] = -x
        lhs[i*2, 4] = -y
        lhs[i*2, 5] = -1
        lhs[i*2, 6] = v * x
        lhs[i*2, 7] = v * y
        lhs[i*2, 8] = v

        lhs[i*2+1, 0] = -x
        lhs[i*2+1, 1] = -y
        lhs[i*2+1, 2] = -1
        lhs[i*2+1, 6] = u * x
        lhs[i*2+1, 7] = u * y
        lhs[i*2+1, 8] = u

    return lhs
