import numpy as np
import cv2
from lmfit import minimize, Parameters
from common import null
from homography import Homography

import pdb


# Camera container
class Camera:
    # Initialize camera with sets of correspondences
    def __init__(self, srcpts, dstpts):
        assert(srcpts.shape[2] == 3)
        assert(dstpts.shape[2] == 2)

        # Store calibration data
        self.srcpts = srcpts
        self.dstpts = dstpts

        (m, n, _) = srcpts.shape
        self.m = m
        self.n = n

        # Calculate H for each image
        hs = np.zeros((m, 3, 3))
        for j, objpts, imgpts in zip(range(m), srcpts, dstpts):
            objpts2d = objpts[:, :2]
            hom = Homography(objpts2d, imgpts)
            # hom.refine()
            hs[j, :, :] = hom.h

        self.hs = hs

        # Estimate intrinsics
        self.K = camera_dlt(hs)
        self.invK = np.linalg.inv(self.K)

        # Extrinsics for each image
        rvecs = np.zeros((m, 3))
        tvecs = np.zeros((m, 3))

        for j, h in zip(range(m), self.hs):
            rvec, tvec = self.extrinsics(h)
            rvecs[j, :] = rvec
            tvecs[j, :] = tvec

        self.rvecs = rvecs
        self.tvecs = tvecs

    # Refine the intrinsics and extrinsics for calibration images
    def refine(self):
        n = self.n
        m = self.m
        K = self.K
        rvecs = self.rvecs
        tvecs = self.tvecs

        srcset = self.srcpts
        dstset = self.dstpts

        params = Parameters()
        add_K(params, K)
        add_rvecs(params, rvecs, m)
        add_tvecs(params, tvecs, m)

        parms = []
        res = minimize(residual, params, args=(srcset, dstset, n, m, parms))

        n = len(parms)
        Ks = np.zeros((n, 3, 3))
        rvecss = np.zeros((n, 3))
        tvecss = np.zeros((n, 3))
        for i, (Kk, rvecc, tvecc) in zip(range(n), parms):
            Ks[i] = Kk
            rvecss[i] = rvecc
            tvecss[i] = tvecc

        np.savez('params_data_data', K=Ks, rvecs=rvecss, tvecs=tvecss)

        self.K = params_to_K(res.params)
        self.invK = np.linalg.inv(self.K)
        self.rvecs = params_to_rvecs(res.params, m)
        self.tvecs = params_to_tvecs(res.params, m)

    # Estimate extrinsics for given homography
    def extrinsics(self, h):
        return camera_extrinsics(self.invK, h)

    # Project world point to sceen space
    def transform(self, rvec, tvec, ms):
        return camera_transform(self.K, rvec, tvec, ms)


def camera_extrinsics(invK, h):
    assert(h.shape == (3, 3))

    # Columns of H
    h1 = h[:, 0]
    h2 = h[:, 1]
    h3 = h[:, 2]

    s = 1/np.linalg.norm(invK @ h1)

    # Extrinsics
    r1 = s * invK @ h1
    r2 = s * invK @ h2
    r3 = np.cross(r1, r2)
    t = s * invK @ h3

    # Convert to vector
    R = np.float32([r1, r2, r3]).T
    rvec, _ = cv2.Rodrigues(R)

    return rvec.reshape(-1), t


# Project world point to sceen space
def camera_transform(K, rvec, tvec, ms):
    assert(K.shape == (3, 3))
    assert(rvec.shape == (3,))
    assert(tvec.shape == (3,))
    assert(ms.shape[1] == 3)

    n = len(ms)

    # Constuct camera matrix C
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.zeros((3, 4))
    Rt[:, :3] = R
    Rt[:, 3] = tvec

    C = K @ Rt

    # Transform each world point
    qs = np.zeros((n, 2))
    for i, m in zip(range(n), ms):
        p = np.ones(4)
        p[:3] = m

        q = C @ p.reshape(-1, 1)
        q /= q[2]

        qs[i, :] = q.reshape(-1)[:2]

    return qs


# Residuals of every correspondences after transformation
def residual(params, srcset, dstset, n, m, parms):
    params.valuesdict()
    K = params_to_K(params)
    rvecs = params_to_rvecs(params, m)
    tvecs = params_to_tvecs(params, m)

    parms.append((K, rvecs[0], tvecs[0]))

    # Calculate the difference between true and estimated image point
    res = np.zeros((m, n, 2))
    for j in range(m):
        clcs = camera_transform(K, rvecs[j], tvecs[j], srcset[j])
        dsts = dstset[j]
        diff = clcs - dsts
        res[j, :, :] = diff

    out = res.reshape(-1)
    return out


# Add intrinsic parameters
def add_K(params, K):
    params.add('a', K[0, 0])
    params.add('c', K[0, 1])
    params.add('u0', K[0, 2])
    params.add('b', K[1, 1])
    params.add('v0', K[1, 2])


# Intrinsic parameters from parameters
def params_to_K(params):
    K = np.zeros((3, 3))
    K[0, 0] = params['a']
    K[0, 1] = params['c']
    K[0, 2] = params['u0']
    K[1, 1] = params['b']
    K[1, 2] = params['v0']
    K[2, 2] = 1
    return K


# Add extrinsic R
def add_rvecs(params, rvecs, m):
    for j, rvec in zip(range(m), rvecs):
        for i, r in zip(range(3), rvec):
            key = 'rvec' + str(j+1) + '_' + str(i+1)
            params.add(key, r)


# Extrinsic R from parameters
def params_to_rvecs(params, m):
    rvecs = np.zeros((m, 3))
    for j in range(m):
        for i in range(3):
            key = 'rvec' + str(j+1) + '_' + str(i+1)
            rvecs[j, i] = params[key]

    return rvecs


# Add extrinsic t
def add_tvecs(params, tvecs, m):
    for j, tvec in zip(range(m), tvecs):
        for i, t in zip(range(3), tvec):
            params.add('tvec' + str(j+1) + '_' + str(i+1), t)


# Extrinsic t from parameters
def params_to_tvecs(params, m):
    tvecs = np.zeros((m, 3))
    for j in range(m):
        for i in range(3):
            tvecs[j, i] = params['tvec' + str(j+1) + '_' + str(i+1)]

    return tvecs


# Helper v_{ij}, see section 2.1 of the report
def vij(h, i, j):
    assert(h.shape == (3, 3))

    v1 = h[0, i]*h[0, j]
    v2 = h[0, i]*h[1, j] + h[1, i]*h[0, j]
    v3 = h[1, i]*h[1, j]
    v4 = h[2, i]*h[0, j] + h[0, i]*h[2, j]
    v5 = h[2, i]*h[1, j] + h[1, i]*h[2, j]
    v6 = h[2, i]*h[2, j]

    return np.float32([v1, v2, v3, v4, v5, v6])


# Estimate camera intrinsics via direct linear transform
def camera_dlt(hs):
    assert(hs.shape[1] == 3)
    assert(hs.shape[2] == 3)

    lhs = camera_lhs(hs)
    b = null(lhs)

    return camera(b)


# Intrinsics from B, see section 2.1 of the report
def camera(b):
    assert(b.shape == (6,))
    b11, b12, b22, b13, b23, b33 = b

    # Intrinsics from Zhang
    v0 = (b12*b13 - b11*b23) / (b11*b22 - b12*b12)
    s = b33 - (b13*b13 + v0*(b12*b13 - b11*b23))/b11
    a = np.sqrt(s/b11)
    b = np.sqrt(s*b11/(b11*b22 - b12*b12))
    c = -b12*a*a*b/s
    u0 = c*v0/a - b13*a*a/s

    # Construct K
    K = np.zeros((3, 3))
    K[0, 0] = a
    K[0, 1] = c
    K[0, 2] = u0
    K[1, 1] = b
    K[1, 2] = v0
    K[2, 2] = 1

    return K


# Construct LHS matrix V
def camera_lhs(hs):
    assert(hs.shape[1] == 3)
    assert(hs.shape[2] == 3)

    n = len(hs)
    assert(n >= 3)

    lhs = np.zeros((2*n, 6))

    for i, h in zip(range(n), hs):
        v12 = vij(h, 0, 1)
        v11 = vij(h, 0, 0)
        v22 = vij(h, 1, 1)
        diff = v11-v22

        lhs[2*i, :] = v12
        lhs[2*i + 1, :] = diff

    return lhs
