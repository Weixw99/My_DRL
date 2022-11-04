"""
在dynamics_model中，new_indeces = npr.randint(0,dataX_new.shape[0], (batchsize_new_pts,))

"""
import numpy as np
import numpy.random as npr

new_indeces = npr.randint(0,5, (3,))

print(new_indeces)
print(type(new_indeces))

# numpy.random.randint(low, high, size=None, dtype='l')
# 从[low,high)中随机生成整数，若high=None，则从[0,low)生成。
# 若size不为none，则按这个size生成整数，这个函数最后产生的是一个数组