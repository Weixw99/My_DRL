"""
numpy.nan_to_num(x):
使用0代替数组x中的nan元素，使用有限的数字代替inf元素
"""
import numpy as np
a = np.array([[np.nan,np.inf],[-np.nan,-np.inf]])
print("a为：",a)

b = np.nan_to_num(a)
print("b为：",b)

"""
输出结果：
a为： [[ nan  inf]
      [ nan -inf]]

b为： [[ 0.00000000e+000  1.79769313e+308]
     [ 0.00000000e+000 -1.79769313e+308]]

"""





