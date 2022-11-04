"""

numpy.clip(a, a_min, a_max, out=None)
其中a是一个数组，后面两个参数分别表示最小和最大值

"""
import numpy as np
x=np.array([1,2,3,5,6,7,8,9])

b = np.clip(x,3,8)
print(b)
print("\n")

# 下面是二维数组
y=np.array([[1,2,3,5,6,7,8,9],[1,2,3,5,6,7,8,9]])
c = np.clip(y,3,8)
print(c)













