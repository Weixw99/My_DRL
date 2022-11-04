"""
main文件中array_meanx = np.tile(np.expand_dims(mean_x, axis=0),(labels_1step.shape[0],1))
第603行
"""
import numpy as np
a = [1, 2, 3,4,5,6,7,8]

w1 = np.array( [a] )
print("w1输出为")
print(w1)
print("\n")

array_meanx1 = np.tile(w1,(5,1)) # (5,1)指将w1的行经过复制，变成原来的5倍，这里是变成5行，列不变
print("array_meanx1输出为")
print(array_meanx1)
print("\n")

# 与上面的不同的是，里面是（5,2）
array_meanx2 = np.tile(w1,(5,2)) # 这里的（5,2）指将w1的行经过复制，变成原来的5倍，这里是变成5行，列经过复制变成原来列的2倍
print("array_meanx2输出为")
print(array_meanx2)


array_meanx3 = np.tile(w1,(5,)) # (5,1)指将w1的行经过复制，变成原来的5倍，这里是变成5行，列不变
print("array_meanx3输出为")
print(array_meanx3)
print(array_meanx3.shape)


"""
输出结果：
w1输出为
[[1 2 3 4 5 6 7 8]]

array_meanx1输出为
[[1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8]]    
 分析：np.tile(w1,(5,1)) ， (5,1)指将w1的行经过复制，变成原来的5倍，这里是变成5行，列不变，最终的size是（5,8）
 

array_meanx2输出为
[[1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8]
 [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8]]

分析：np.tile(w1,(5,2)) ，（5,2）指将w1的行经过复制，变成原来的5倍，这里是变成5行，列经过复制变成原来列的2倍
最终的size是（5,16）

"""