


import numpy as np

a = [1, 2] ; b = [4, 5]
c = [7, 8];d = [11,15];e = [20,30];f = [0,0]
k = [33,44];l = [99,77];m = [11,22]
w1 = np.array([[a,b,c],[d,e,f],[k,l,m]])

print("w1输出为",w1)

print("******************************")

# 这里才是重点
length_curr_rollout = 6
list_100 = []
for i in range(3):
    for j in range(4):
        list_100.append(w1[i][0+j:length_curr_rollout-1+j])

        print(list_100)
        print("哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈，用于隔开")

print('\n')
print("最终的list_100：",list_100)




