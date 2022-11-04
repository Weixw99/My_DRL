
"""
old_indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)


#### numpy.random.choice(a, size=None, replace=False, p=None)
#  意思是从a中以概率P，随机选择size个
# 比如：a = np.random.choice(a=5, size=3, replace=False, p=None)
# 表示：从a 中以概率P，随机选择3个,从0开始选，取不到a，最多取到a-1. p没有指定的时候相当于是一致的分布
# 结果可能是：[3 1 4]或者[4 1 0]等等......
# a: 采样的样本，为一维数组或整型变量         size: 采样的大小
#### replace: 采样是否有放回（重复使用数据），False表示不放回    p: 采样概率，None表示均匀采样


"""
import numpy as np
import numpy.random as npr

a = np.array([1, 2,11,15,19])

old_indeces = npr.choice(a, size=(5,), replace=False)

print(old_indeces)
