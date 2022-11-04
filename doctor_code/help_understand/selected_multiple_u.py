
import numpy as np

a = [1, 2] ; b = [4, 5]
c = [7, 8];d = [11,15];e = [20,30];f = [0,0]
k = [33,44];l = [99,77];m = [11,22]
w1 = np.array([[a],[d],[k],[e],[f],[m]])
print("w1输出为")
print(w1)

print("w1.shape",w1.shape)


selected_multiple_u=[]
for rollout_num in range(10):

    selected_multiple_u.append(w1)

print(selected_multiple_u)

selected_multiple_u_array = np.array(selected_multiple_u)
print("selected_multiple_u_array.shape为：",selected_multiple_u_array.shape)

w2 = np.squeeze(np.array(selected_multiple_u), axis=2)[0:(4)]

print("w2:",w2)

print(w2.shape)