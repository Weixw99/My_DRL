
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

list = []
for i in range(3):
    list.append(w1[i][0:3])

    print(list)
#    listarray = np.copy(list)
    print("####################")
#    print(listarray)
#    print("####################")

list = np.concatenate(list)

print("故意隔开####################")

print(list)   # 输出是2维数组
print("\n")

print("list[0][0]输出为") # 2维数组[][]输出的是一个数字
print(list[0][0])

