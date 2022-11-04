
"""

在dynamics_model中，大约216行：
dataX_batch = np.concatenate((dataX_old_batch, dataX_new_batch))
dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))
"""

import numpy as np

a = [1, 2, 3] ; b = [4, 5, 6]
w1 = np.array([a,b])
print("w1输出为")
print(w1)
print("\n")

"""
w1输出：
[[1 2 3]
 [4 5 6]]
"""

c = [7, 8, 9];d = [11,15,19]
w2 = np.array([c,d])
print("w2输出为")
print(w2)
print("\n")
"""
w2输出：
[[ 7  8  9]
 [11 15 19]]
"""

# e = np.concatenate((w1,w2),axis=0) 这个跟下面这个没有axis=0的一样
e = np.concatenate((w1,w2))
print("***********")
print("e输出为")
print(e)
print(type(e))
print("***********")
print("\n")
"""
e输出：
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [11 15 19]]
 看出，结果是让行数变多了
 类型：<class 'numpy.ndarray'>
"""

f = np.concatenate((w1,w2),axis=1)
print("f输出为")
print(f)

m=np.concatenate(f,axis=0)

print("m输出为：",m)

"""
f输出：
[[ 1  2  3  7  8  9]
 [ 4  5  6 11 15 19]]
 从上面看出，np.concatenate((w1,w2),axis=1)是拼接列，让每一行的数据变多，也就是列数多了
"""
