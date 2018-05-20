import numpy as np
import cv2
from calibration import camera_transform


with np.load('params_data_data.npz') as X:
    Ks, rvecs, tvecs = [X[i] for i in ('K', 'rvecs', 'tvecs')]


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


source = cv2.imread('left/left13.jpg')

axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

images = []
for K, rvec, tvec in zip(Ks, rvecs, tvecs):
    pts = camera_transform(K, rvec, tvec, axis)
    nextimg = source.copy()
    nextimg = draw(nextimg, pts)
    images.append(nextimg)

n = len(images)

cv2.namedWindow('Animate')
cv2.setWindowProperty('Animate', cv2.WINDOW_AUTOSIZE, 1)
cv2.setWindowProperty('Animate', cv2.WINDOW_FULLSCREEN, 1)

while(1):
    print('restart')
    for i in range(500):
        print('Frame {}'.format(i))
        cv2.imshow('Animate', images[i])
        cv2.waitKey(1) & 0xFF


cv2.destroyAllWindows()
