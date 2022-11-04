"""
np.linspace()的用法：
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

在规定的时间内，返回固定间隔的数据。他将返回“num”个等间距的样本，在区间[start, stop]中。
其中，区间的结束端点可以被排除在外。

分析：
start：队列的开始值
stop：队列的结束值
num：要生成的样本数，非负数，默认是50
endpoint：若为True，“stop”是最后的样本；否则“stop”将不会被包含。默认为True
retstep：若为False，返回等差数列；否则返回array([samples, step])。默认为False
"""

from matplotlib import pyplot as plt

import numpy as np

x = np.linspace(-np.pi, np.pi, 256, endpoint=True) # 从-π到π，共256个值，包括π

# 两个三角函数，也有256个值
y = np.sin(x)
#z = np.cos(x)

plt.plot(x, y)

# plt.plot(x, z)

plt.show()



