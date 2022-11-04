"""
在dynamics_model中，range_of_indeces = np.arange(dataX.shape[0])

"""
import numpy as np

#一个参数 ，默认起点0，步长为1 输出：[0 1 2]
a = np.arange(3)

#两个参数， 默认步长为1 输出[3 4 5 6 7 8]
a = np.arange(3,9)

#三个参数 ，起点为0，终点为2，步长为0.1
# 输出[ 0. 0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.  1.1  1.2  1.3  1.4 1.5  1.6  1.7  1.8  1.9  2.]
a = np.arange(0, 2, 0.1)


