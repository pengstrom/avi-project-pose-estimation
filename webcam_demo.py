import numpy as np
import cv2
from homography import Homography
from calibration import camera_transform, camera_extrinsics

dest = 'webcam_calib'

cap = cv2.VideoCapture(0)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

with np.load('{}/webcam.npz'.format(dest)) as X:
    K = X['K']

invK = np.linalg.inv(K)


objp = np.zeros((7*7, 2), np.float32)
objp = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


while(1):
    ret, frameflip = cap.read()

    frame = cv2.flip(frameflip, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 7),
                                             flags=(cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE))

    if ret:
        # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
        #                             criteria)

        corners2 = corners

        dstpts = corners2.reshape(-1, 2)

        hom = Homography(objp, dstpts)

        rvec, tvec = camera_extrinsics(invK, hom.h)

        imgpts = camera_transform(K, rvec, tvec, axis)

        img = draw(frame, imgpts)

    cv2.imshow('Demo', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print('Exiting')
        break

cap.release()
cv2.destroyAllWindows()
