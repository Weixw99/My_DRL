"""
main程序里看看下面这个是怎么回事
list_100 = []
for j in range(100):
    list_100.append(controls_val[i][0+j:length_curr_rollout-100+j])

list_100=np.array(list_100) #100xstepsx2  ？？？
list_100= np.swapaxes(list_100,0,1)
"""
import numpy as np

a = [1, 2, 3] ; b = [4, 5, 6]
c = [7, 8, 9];d = [11,15,19];e = [20,30,40];f = [90,33,22]
w1 = np.array([a,b,c,d,e,f])
print("w1输出为")
print(w1)
print("\n")


list = []
for j in range(5):
    list.append(w1[0+j:1+j])
    print(list) # 查看与下面相邻的print(list)输出的不同
    print("\n")

    listarray = np.copy(list)
    print("####################")
    print(listarray)  # 这里输出就是3维数组了
    print("####################")

print("\n")

print("隔开*******************")
print(list)
print("*******************")


