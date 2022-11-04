

import numpy as np

a = [1, 2] ; b = [4, 5]
c = [7, 8];d = [11,15];e = [20,30];f = [0,0]
k = [33,44];l = [99,77];m = [11,22]
w1 = np.array([[a,b,c],[d,e,f],[k,l,m]])

print("w1输出为")
print(w1)


differences=[]
for states_in_single_rollout in w1:

    differences.append(states_in_single_rollout)



print("differences输出为",differences)
print("differences类型为",type(differences))  # 列表
#print("differences.shape为",differences.shape)  # 这句话数错的，AttributeError: 'list' object has no attribute 'shape'

new_differences = np.array(differences)
print("new_differences输出为",new_differences)
print("new_differences.shape为",new_differences.shape)  # 现在是3维数组


output = np.concatenate(differences, axis=0)
print("output输出为",output)   # 2维数组
print("output.shape输出为",output.shape)



