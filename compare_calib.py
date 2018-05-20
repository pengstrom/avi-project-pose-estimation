import cv2
import numpy as np
import glob
from calibration import Camera

# Load previously saved data
with np.load('camera.npz') as X:
    cv_mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def draw(img, imgpts, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    output = img.copy()
    overlay = img.copy()

    # draw ground floor in green
    cv2.drawContours(overlay, [imgpts[:4]], -1, color, -3)
    cv2.addWeighted(overlay, 0.60, output, 0.40, 0, output)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(overlay, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)
        cv2.addWeighted(overlay, 0.60, output, 0.40, 0, output)

    # draw top layer in red color
    cv2.drawContours(overlay, [imgpts[4:]], -1, color, 3)
    cv2.addWeighted(overlay, 0.60, output, 0.40, 0, output)

    return output


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

objpts = []
imgpts = []
imgs = []

images = glob.glob('left/left*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        print(fname)
        objpts.append(objp)
        imgs.append(img)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria)
        imgpts.append(corners2.reshape(-1, 2))

# Calibrate camera and refine
camera = Camera(np.array(objpts), np.array(imgpts))

np.savez('camera_calib_unrefined', K=camera.K, invK=camera.invK, rvecs=camera.rvecs, tvecs=camera.tvecs)

camera.refine()

np.savez('camera_calib_refined', K=camera.K, invK=camera.invK, rvecs=camera.rvecs, tvecs=camera.tvecs)

camera.quit()

print('Intrinsics')
print('OpenCV intrinsics:')
print(cv_mtx)
print('My intrinsics:')
print(camera.K)
print('Error:')
print(camera.K - cv_mtx)

print('Extrinsics')
for objp, imgp, img, h, rvec, tvec in zip(objpts, imgpts, imgs, camera.hs,
                                          camera.rvecs, camera.tvecs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, cv_rvec, cv_tvec, _ = cv2.solvePnPRansac(objp, imgp,
                                                cv_mtx, dist)

    print('Opencv rvec:')
    print(cv_rvec.reshape(-1))
    print('My rvec:')
    print(rvec)
    print('Error:')
    print(rvec - cv_rvec.reshape(-1))
    print('Opencv tvec:')
    print(cv_tvec.reshape(-1))
    print('My tvec:')
    print(tvec)
    print('Error:')
    print(tvec - cv_tvec.reshape(-1))

    # project 3D points to image plane using OpenCV
    cv_imgpts, jac = cv2.projectPoints(axis, cv_rvec, cv_tvec, cv_mtx, dist)
    
    # project 3D points to image plane using my implementation
    my_imgpts = camera.transform(rvec, tvec, axis)

    # Draw test cube
    img = draw(img, cv_imgpts, (255,0,0))
    img = draw(img, my_imgpts, (0,255,0))
    cv2.imshow('img', img)
    k = cv2.waitKey(0) & 0xff
    if k == 's':
        cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()
