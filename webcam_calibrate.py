import numpy as np
import cv2
from calibration import Camera
from glob import glob

dest = 'webcam_calib'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

objlist = []
imglist = []

images = glob('{}/*.png'.format(dest))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        objlist.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria)

        imglist.append(corners2.reshape(-1, 2))

print('Found {} good images out of {}!'.format(len(objlist), len(images)))

objset = np.array(objlist)
imgset = np.array(imglist)

print('Saving data...')
np.savez('chess_data', objset=objset, imgset=imgset)

camera = Camera(objset, imgset)
camera.refine()

np.savez('{}/webcam'.format(dest), K=camera.K)
