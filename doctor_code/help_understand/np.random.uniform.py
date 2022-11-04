"""
numpy.random.uniform(low=0.0, high=1.0, size=None)
从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high，没有任何参数的话，是从[0, 1)
"""

import numpy as np

#x7 = np.random.uniform(-2,2,1)
x7 = np.random.uniform(-2,2,(4,3,1))

print(x7)
print(x7.dtype)

"""
分析：
从一个均匀分布[-2,2)中随机采样，采样后的数据以3维形式显示
（4,3,1）的意思是一个3维矩阵
4：三维矩阵里面有4个二维数组元素
3:每一个二维数组都有3个一维数组元素
1：一维数组里面只有一个数值元素

"""



