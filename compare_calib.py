import cv2
import numpy as np
import glob
from calibration import Camera
import matplotlib

# Load previously saved data
with np.load('camera.npz') as X:
    cv_mtx, dist, cv_rvecs, cv_tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


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
        objpts.append(objp)
        imgs.append(img)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria)
        imgpts.append(corners2.reshape(-1, 2))

camera = Camera(np.array(objpts), np.array(imgpts))

# camera.K = cv_mtx
# camera.invK = np.linalg.inv(cv_mtx)

# camera.tvecs = cv_tvecs.reshape(-1, 3)
# camera.rvecs = cv_rvecs.reshape(-1, 3)

tvec_diff = [x/y for x, y in zip(camera.tvecs.reshape(-1), cv_tvecs.reshape(-1))]
avgmean = np.mean(tvec_diff)
# camera.tvecs /= avgmean
# matplotlib.pyplot.bar(range(len(tvec_diff)), tvec_diff)
# matplotlib.pyplot.show()

camera.refine()

print('Intrinsics')
print('OpenCV intrinsics:')
print(cv_mtx)
print('My intrinsics:')
print(camera.K)
print('Error:')
print(camera.K - cv_mtx)

print(camera.s)

print('Extrinsics')
for objp, imgp, img, h, rvec, tvec in zip(objpts, imgpts, imgs, camera.hs,
                                          camera.rvecs, camera.tvecs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, cv_rvec, cv_tvec, _ = cv2.solvePnPRansac(objp, imgp,
                                                cv_mtx, dist)

    print('Opencv rvec:')
    print(cv_rvec)
    print('My rvec:')
    print(rvec)
    print('Error:')
    print(rvec - cv_rvec.reshape(-1))
    print('Opencv tvec:')
    print(cv_tvec)
    print('My tvec:')
    print(tvec)
    print('Error:')
    print(tvec - cv_tvec.reshape(-1))

    # project 3D points to image plane
    cv_imgpts, jac = cv2.projectPoints(axis, cv_rvec, cv_tvec, cv_mtx, dist)
    
    my_imgpts = camera.transform(rvec, tvec, axis)

    # img = draw(img, corners2, cv_imgpts)
    img = draw(img, corners2, my_imgpts)
    cv2.imshow('img', img)
    k = cv2.waitKey(0) & 0xff
    if k == 's':
        cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()
