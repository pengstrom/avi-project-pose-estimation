import numpy as np
import cv2
from common import null


class Camera:
    def __init__(self, hs):
        self.hs = hs
        self.A, self.s = camera_dlt(hs)
        self.Ainv = np.inv(self.A)

    def refine(self):
        pass

    def extrinsics(self, h):
        h1 = h[:, 0]
        h2 = h[:, 1]
        h3 = h[:, 2]

        r1 = self.la * self.invA @ h1
        r2 = self.la * self.invA @ h2
        r3 = np.cross(r1, r2)
        t = self.la * self.invA @ h3

        # The [R t] 3x4 matrix
        R = np.float32([r1, r2, r3]).T
        rvecs = cv2.Rodrigues(R)

        return rvecs, t


def vij(h, i, j):
    v1 = h[0, i]*h[0, j]
    v2 = h[0, i]*h[1, j] + h[1, i]*h[0, j]
    v3 = h[1, i]*h[1, j]
    v4 = h[2, i]*h[0, j] + h[0, i]*h[2, j]
    v5 = h[2, i]*h[1, j] + h[1, i]*h[2, j]
    v6 = h[2, i]*h[2, j]
    return np.float32([v1, v2, v3, v4, v5, v6])


def camera_dlt(hs):
    lhs = camera_lhs(hs)
    b = null(lhs)

    return camera(b)


def camera(b):
    b11, b12, b22, b13, b23, b33 = b

    # Intrinsics from Zhang
    v0 = (b12*b13 - b11*b23) / (b11*b22 - b12*b12)
    la = b33 - (b13*b13 + v0*(b12*b13 - b11*b23))/b11
    a = np.sqrt(la/b11)
    b = np.sqrt(la*b11/(b11*b22 - b12*b12))
    c = -b12*a*a*b/la
    u0 = c*v0/a - b13*a*a/la

    A = np.zeros(3, 3)
    A[0, 0] = a
    A[0, 1] = c
    A[0, 2] = u0
    A[1, 1] = b
    A[1, 2] = v0
    A[2, 2] = 1

    return A, la


def camera_lhs(hs):
    n = len(hs)
    assert(n >= 3)

    lhs = np.zeros(2*n, 6)

    for i, h in zip(range(n), hs):
        v12 = vij(h, 0, 1)
        v11 = vij(h, 0, 0)
        v22 = vij(h, 1, 1)
        diff = v11-v22

        for j in range(6):
            lhs[2*i, j] = v12[j]
            lhs[2*i + 1, j] = diff[j]

    return lhs


