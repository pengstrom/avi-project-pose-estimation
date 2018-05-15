import numpy as np
import cv2
# from termcolors import cprint
from homography import Homography


# def bprint(x):
#     return cprint(x, attrs=['bold'])

# IMGSIZE = 512
# SCALE = 0.8


# def draw_poly(img, pts, color):
#    vs = np.zeros((len(pts), 2), np.int32)
#    for v, p in zip(vs, pts):
#        v[0] = np.round(pts[0]*IMGSIZE*SCALE/2 + IMGSIZE/2)
#        v[1] = np.round(pts[1]*IMGSIZE*SCALE/2 + IMGSIZE/2)
#    vs = vs.reshape(-1, 1, 2)
#    cv2.polylines(img, vs, True, color)


# def draw_img(srcpts, dstpts, h, h_cv):
#    img = np.zeros((IMGSIZE, IMGSIZE, 3), np.uint8)
#    draw_poly(img, srcpts, (255, 0, 0))
#    draw_poly(img, dstpts, (0, 255, 0))


print('Homography test')

# Source points
print('Source points')
srcpts = np.float64([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])
print(srcpts)


# Destination points
print('Destination points')
dstpts = np.float64([[-0.5, -0.5], [0.52, -0.4],
                     [0.5, 0.5], [-0.5, 0.5], [0, 0]])
for dst in dstpts:
    dst[0] = np.round(dst[0] + np.random.normal(scale=0.05), 2)
    dst[1] = np.round(dst[1] + np.random.normal(scale=0.05), 2)

# cprint(dstpts, 'green')
print(dstpts)

h_cv, _ = cv2.findHomography(srcpts, dstpts)

hgraf = Homography(srcpts, dstpts)
hgraf.refine()
h = hgraf.h

# cv2.imshow(draw_img(srcpts, dstpts, h, h_cv))

print("OpenCV solution:")
# cprint(h_cv, 'yellow')
print(h_cv)

print("My solution:")
print(h)

print("Difference:")
# cprint(h-h_cv, 'red')
print(h-h_cv)
