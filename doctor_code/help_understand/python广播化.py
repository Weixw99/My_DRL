"""
在dynamics_model里面，第436行
# self.mean_y是1维的，(1,2)，而curr_control是3维的，发生python广播机制，进行相减，
# 输出的curr_control_preprocessed是3维的，（20,2）
curr_control_preprocessed = curr_control - self.mean_y

"""
import numpy as np

mean_y =np.array([1,2])

print("1维数组mean_y:")
print(mean_y)
print("\n")

a = [3,4];b = [5,6];c = [7,8]
curr_control = np.array([[a,b,c]])

print("3维数组curr_control：")
print(curr_control)
print("\n")

curr_control_preprocessed = curr_control - mean_y  # 这里发生python广播化

print("二者相减结果：")
print(curr_control_preprocessed)  # 看最后结果
