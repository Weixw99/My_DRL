"""

3维数组的索引
"""

import numpy as np

a = [1, 2] ; b = [4, 5]
c = [7, 8];d = [11,15];e = [20,30];f = [0,0]
k = [33,44];l = [99,77];m = [11,22]
w1 = np.array([[a,b,c],[d,e,f],[k,l,m]])

print("w1输出为")
print(w1)

x_array = np.array(w1)[0:2] # 这是main的839行，x_array = np.array(resulting_multiple_x)[0:(rollouts_forTraining+1)]
print("x_array输出为",x_array)
print("x_array.shape输出为",x_array.shape)


print("w1[0]输出为",w1[0])
print("w1[1]输出为",w1[1])

print("len[w1]输出为",len(w1))  # 3维数组w1的里面2维数组的个数

print("w1[0].shape[0]输出为",w1[0].shape[0])

print("w1[:,1,:]输出为：",w1[:,1,:])
print("\n")



print("w1[:,0,:]为",w1[:,0,:])  # 这里组成了一个2维数组

for curr_control in w1:
    #print("curr_control为", curr_control)

    curr_control = np.expand_dims(curr_control, axis=0)

    print("现在的curr_control为", curr_control)