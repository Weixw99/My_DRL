"""
在dynamics_model中，大约192行：
dataX_new_batch = dataX_new[new_indeces, :]
dataZ_new_batch = dataZ_new[new_indeces, :]

"""
import numpy as np

a = [1, 2, 3] ; b = [4, 5, 6]; c = [7, 8, 9];d = [6,6,6]

dataX_new = np.array( [a,b,c,d] )
print("dataX_new输出为：")
print(dataX_new)
print("\n")

e = [0,1,2]
new_indeces = np.array(e)
print(new_indeces)
print(type(new_indeces))
print("\n")

dataX_new_batch1 = dataX_new[new_indeces,:]   # 这个地方注意一下,与dataX_new_batch = dataX_new[new_indeces]相同
print(dataX_new_batch1)  # 输出的是2维数组

# 其他例子;
print("\n")
dataX_new_batch2 = dataX_new[new_indeces,2]  # 显示的是在上面显示的基础上，输出第2列数据
print(dataX_new_batch2) # 现在显示的是1维数组

