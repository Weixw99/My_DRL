

import numpy as np
a = [1, 2] ; b = [4, 5]; c = [7, 8]
w1 = np.array( [a,b,c] )
print(w1)

curr_control = np.expand_dims(w1, axis=0)
print("curr_control为：",curr_control)
print("curr_control.shape为：",curr_control.shape)   # (1,3,2)

d = np.array([9,9,9,9])
print(d)

e =np.append(w1,d)
print(e)


f = np.expand_dims(e,axis=0)
print(f)
print(f.shape)

