import numpy as np
import matplotlib.pyplot as plt

with np.load('compare_hom2.npz') as X:
    y = [float(X[i]) for i in ('cv_mean', 'my_mean', 'my_refined_mean')]

x = [0, 1, 2]
labels = ['OpenCV', 'Unrefined', 'Refined']

print(x)
print(y)
print(labels)

plt.bar(x, y, tick_label=labels, width=0.9)
plt.show()
