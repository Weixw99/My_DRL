
"""

在dynamics-model.py文件中，第253行：
p = npr.permutation(dataX_new.shape[0])

"""

import numpy as np
a = np.array([[1 ,2 ,3 ,4] ,[5 ,3 ,4 ,2] ,[0 ,5 ,8 ,9] ,[0 ,4 ,5 ,2] ,[3 ,0 ,1 ,6]])
print("a输出为：")
print(a)
print("\n")

b = a.shape[0]
print("b输出为：")
print(b)
print("\n")

c = np.random.permutation(b)  #
print("c输出为：")
print(c)
print("\n")

print("a[c]输出为：")
print(a[c])  # 这个才是最后输出的

"""
a输出为：
[[1 2 3 4]
 [5 3 4 2]
 [0 5 8 9]
 [0 4 5 2]
 [3 0 1 6]]

b输出为：
5

c输出为：
[2 3 1 4 0]    

a[c]输出为：
array([[0, 5, 8, 9],
       [0, 4, 5, 2],
       [5, 3, 4, 2],
       [3, 0, 1, 6],
       [1, 2, 3, 4]])
       
注意：c输出为：[2 3 1 4 0]，看里面数字元素2，将原数组的第二行（从第0行开始计数）放到新数组的第0行

"""

