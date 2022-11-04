
import numpy as np

a = [1, 2, 3] ; b = [4, 5, 6]
c = [7, 8, 9];d = [11,15,19];e = [20,30,40];f = [0,0,0]
k = [33,44,55];l = [99,77,88];m = [11,22,00]
w1 = np.array([[a,b,c],[d,e,f],[k,l,m]])

print("w1输出为")
print(w1)
print("\n")

"""
w1输出：
[[[ 1  2  3]
  [ 4  5  6]
  [ 7  8  9]]

 [[11 15 19]
  [20 30 40]
  [ 0  0  0]]

 [[33 44 55]
  [99 77 88]
  [11 22  0]]]

"""

controls_100step=[]


for i in range(3):

    list = []
    for j in range(3):
    #    print(w1[0][0:2])
    #    print("注意看看")

        list.append(w1[i][j:1+j])

        print(list)
        print("\n")
    #    listarray = np.copy(list)
    #    print("####################")
    controls_100step.append(list)
    #    print(listarray)
    #    print("####################")

controls_100step = np.concatenate(controls_100step)

print("故意隔开便于查看下面输出")

print(controls_100step)   # 输出是3维数组

print(controls_100step.shape)





