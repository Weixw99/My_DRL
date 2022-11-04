"""
在dynamics_model中，第419行的循环：
 for curr_control in forwardsim_y:
 其中，forwardsim_y是3维数组

"""
import numpy as np

a = [1, 2, 3] ; b = [4, 5, 6]; c = [7, 8, 9];d = [10,11,12];e = [13,14,15];f = [16,16,16]

w1 = np.array([[a,b,c],[d,e,f]])

print(w1)  # w1是一个3维数组
print("哈哈哈哈哈")



for curr_control in w1:

    curr_control = np.expand_dims(curr_control, axis=0)

    print(curr_control)   # 这里输出的是3维数组，是将里面每一个2维数组作为元素，在输出时变成3维
    print("有没有分开。。。。。。。。。。")

print("\n")

print(curr_control)  # 这里输出的curr_control是上面循环最后一次输出的一个3维数组，也是w1里面的最后1个2维数组变成了3维数组