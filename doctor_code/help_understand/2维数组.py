import numpy as np

a = [13, 21, 30,11,15,19] ; b = [4, 5, 6,20,30,40]
c = [7, 8, 999,77,88,777];d = [0,0,0,3,4,2]

h=[1,1];j=[2,2];k=[3,3];l=[4,4]
eta =  np.array([[9], [6], [33]])

print("eta.shape",eta.shape)

w1 = np.array([a,b,c,d])

e=[1,2]

e=np.array(e)

w2 = np.array([h,j,k,l])

squ=np.squeeze(w2)
print("w2为：",w2)
print("squ为：",squ)



print("w1输出为:")
print(w1)
print("\n")
print("w1.shape为",w1.shape)
print("w1[0][0]为",w1[0][0])

print("w1[e,:]为",w1[e,:])

print("w1[0:-1,:]为",w1[0:-1,:])   # 去除w1的最后一行
#x[0:-1,:]

print("w1[1:, :]为",w1[1:, :])





print("w1.shape[0]为",w1.shape[0])
print("w1[0]为",w1[0])
print("len(w1)为",len(w1))
print(w1[:,3])

e = np.array([c])
print("e输出为:",e)

print("e[0][0]输出为:",e[0][0])





