
"""
main里面第675行
 error_1step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[1]-array_meanx,array_stdx))
                                -np.nan_to_num(np.divide(labels_1step-array_meanx,array_stdx))))


"""

import numpy as np

a = [1, 2, 3,11] ; b = [4, 5, 6,20]
c = [7, 8, 999,77]
w1 = np.array([a,b,c])

print("w1输出为:")
print(w1)

d = [11,15,10,19];e = [20,30,100,40];f = [0,0,88,777]
w2 = np.array([d,e,f])

print("w2输出为:")
print(w2)

k = [1,1,1,1];l = [1,1,1,1];m = [1,1,1,1]
array_meanx = np.array([k,l,m])

print("array_meanx输出为:")
print(array_meanx)


o = [3,3,3,3];p = [3,3,3,3];q = [3,3,3,3]
array_stdx = np.array([o,p,q])

print("array_stdx输出为:")
print(array_stdx)


error_1step = np.mean(np.square(np.nan_to_num(np.divide(w1 - array_meanx,array_stdx))
                                -np.nan_to_num(np.divide(w2-array_meanx,array_stdx))))

print(error_1step)
print(error_1step.shape)
