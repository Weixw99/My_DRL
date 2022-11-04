"""
reshape的训练，防止感觉模糊

"""
import numpy as np

a = [1, 2, 3,4] ; b = [5, 6,7,8]; c = [9,10,11,12]
m = np.array([a,b,c])
print(m)
print("\n")

n = m.reshape(m.shape[0],-1).T

print(n)