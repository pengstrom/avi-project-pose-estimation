import numpy as np
import cv2
# from termcolors import cprint
from homography import Homography, transform

with np.load('chess_data.npz') as X:
    objset, imgset = [X[i] for i in ('objset', 'imgset')]


(m, n, _) = objset.shape

cv_errs = np.zeros((m, n))
my_errs = np.zeros((m, n))
my_refined_errs = np.zeros((m, n))

objlist = np.zeros((m, n, 2))
objlist = objset[:, :, :2]

for k, srcpts, dstpts in zip(range(m), objlist, imgset):
    h_cv, _ = cv2.findHomography(srcpts, dstpts)
    cv_dst = cv2.perspectiveTransform(srcpts.reshape(-1, 1, 2), h_cv)
    diff = cv_dst.reshape(-1, 2) - dstpts
    cv_errs[k] = np.linalg.norm(diff, axis=1)

    hgraf = Homography(srcpts, dstpts)
    for i, src, dst in zip(range(n), srcpts, dstpts):
        my_dst = transform(src, hgraf.h)
        my_errs[k, i] = np.linalg.norm(my_dst - dst)

    hgraf.refine()
    for i, src, dst in zip(range(n), srcpts, dstpts):
        my_dst = transform(src, hgraf.h)
        my_refined_errs[k, i] = np.linalg.norm(my_dst - dst)


cv_mean = np.mean(cv_errs.reshape(-1))
cv_std = np.std(cv_errs.reshape(-1))

my_mean = np.mean(my_errs.reshape(-1))
my_std = np.std(my_errs.reshape(-1))

my_refined_mean = np.mean(my_refined_errs.reshape(-1))
my_refined_std = np.std(my_refined_errs.reshape(-1))

print('Homography evaluation')

print('\nOpenCV\n')

print('Mean error:')
print(cv_mean)

print('\nUnrefined\n')

print('Mean error:')
print(my_mean)

print('\nRefined\n')

print('Mean error:')
print(my_refined_mean)

np.savez('compare_hom2', cv_mean=cv_mean, cv_std=cv_std, my_mean=my_mean, my_std=my_std, my_refined_mean=my_refined_mean, my_refined_std=my_refined_std)
