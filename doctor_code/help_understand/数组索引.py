"""
在dynamics_model中，大约209行：
dataX_old_batch = dataX[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]

"""
import numpy as np

a = [1, 2, 3] ; b = [4, 5, 6]; c = [7, 8, 9];d = [6,6,6];f = [7,7,7];g = [8,8,8];h = [9,9,9]

dataX = np.array( [a,b,c,d,f,g,h] )
print("dataX输出为：")
print(dataX)
print("\n")

m = [0,1,2,3,4,5,6,7,8,9,10]

old_indeces = np.array(m)
print(old_indeces)
print("\n")

batch = 1
batchsize_old_pts = 3

t = old_indeces[1*3:2*3]
print(t)   # 这里输出[3 4 5]
print("\n")

dataX_old_batch = dataX[old_indeces[1*3:2*3], :]  # 这里

print(dataX_old_batch)  # 这里输出
"""
这里输出
[[6 6 6]
 [7 7 7]
 [8 8 8]]
"""