
import numpy as np

a = [1, 11, 6,11,15,19]

w1 = np.array(a)

print("数组a：",a)
print("w1.shape",w1.shape)

w2=[w1,0]

print("w2：",w2)
print("w2.shape",w2.shape)  # 这个是错误的，w2是列表

print("数组a[2]：",a[2])

