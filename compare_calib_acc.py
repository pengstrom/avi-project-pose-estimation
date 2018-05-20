import numpy as np
import cv2

with np.load('camera.npz') as X:
    cv = [X[i] for i in ('mtx', 'rvecs', 'tvecs')]

with np.load('camera_calib_unrefined.npz') as X:
    unrefined = [X[i] for i in ('K', 'rvecs', 'tvecs')]

with np.load('camera_calib_refined.npz') as X:
    refined = [X[i] for i in ('K', 'rvecs', 'tvecs')]

(m, _) = unrefined[1].shape

accuracy = np.zeros((3, m, m, 2, 3))

for k, method in zip(range(3), [cv, unrefined, refined]):
    for j1, rvec1, tvec1 in zip(range(m), method[1], method[2]):
        for j2, rvec2, tvec2 in zip(range(j1), method[1], method[2]):
            accuracy[k, j1, j2, 0] = rvec1.reshape(-1) - rvec2.reshape(-1)
            accuracy[k, j1, j2, 1] = tvec1.reshape(-1) - tvec2.reshape(-1)

cv_my = accuracy[0] - accuracy[1]
my_ref = accuracy[1] - accuracy[2]
cv_ref = accuracy[0] - accuracy[2]


tvec_cv_my = np.linalg.norm(cv_my[:, :, 1], axis=2)
nonzero = tvec_cv_my != 0

tvec_cv_my_vec = np.sum(cv_my[nonzero], axis=(0, 1))/len(cv_my[nonzero])
print(tvec_cv_my_vec)

tvec_cv_my_mean = np.mean(tvec_cv_my[nonzero])
tvec_cv_my_std = np.std(tvec_cv_my[nonzero])


tvec_my_ref = np.linalg.norm(my_ref[:, :, 1], axis=2)
nonzero = tvec_my_ref != 0
tvec_my_ref_mean = np.mean(tvec_my_ref[nonzero])
tvec_my_ref_std = np.std(tvec_my_ref[nonzero])

tvec_cv_ref = np.linalg.norm(cv_ref[:, :, 1], axis=2)
nonzero = tvec_cv_ref != 0
tvec_cv_ref_mean = np.mean(tvec_cv_ref[nonzero])
tvec_cv_ref_std = np.std(tvec_cv_ref[nonzero])

np.set_printoptions(precision=4, linewidth=400)

print('Comparison between translation vector t between methods and 11 images.')

print('\nOpenCV vs My method:\n')
print(tvec_cv_my)
print('Mean:')
print(tvec_cv_my_mean)
print('Std:')
print(tvec_cv_my_std)
print('Confidence interval:')
print((tvec_cv_my_mean - tvec_cv_my_std * 1.96 / np.sqrt(len(tvec_cv_my[nonzero])), tvec_cv_my_mean + tvec_cv_my_std * 1.96 / np.sqrt(len(tvec_cv_my[nonzero]))))

print('\nMy method vs My refined method:\n')
print(tvec_my_ref)
print('Mean:')
print(tvec_my_ref_mean)
print('Std:')
print(tvec_my_ref_std)
print('Confidence interval:')
print((tvec_my_ref_mean - tvec_my_ref_std * 1.96 / np.sqrt(len(tvec_my_ref[nonzero])), tvec_my_ref_mean + tvec_my_ref_std * 1.96 / np.sqrt(len(tvec_my_ref[nonzero]))))

print('\nOpenCV vs My refined method:\n')
print(tvec_cv_ref)
print('Mean:')
print(tvec_cv_ref_mean)
print('Std:')
print(tvec_cv_ref_std)
print('Confidence interval:')
print((tvec_cv_ref_mean - tvec_cv_ref_std * 1.96 / np.sqrt(len(tvec_cv_ref[nonzero])), tvec_cv_ref_mean + tvec_cv_ref_std * 1.96 / np.sqrt(len(tvec_cv_ref[nonzero]))))

np.savez('camera_calib_accuracy', accuracy=accuracy)
