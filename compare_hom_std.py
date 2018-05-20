import numpy as np
import matplotlib.pyplot as plt

with np.load('compare_hom2.npz') as X:
    cv, my, my_refined = [float(X[i]) for i in ('cv_std', 'my_std', 'my_refined_std')]


print('CV')
print(cv)

print('My')
print(my)

print('ref')
print(my_refined)
