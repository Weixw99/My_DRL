import math as mt
import numpy as np
x = -1
y = -1
ex = 0
ey = 0

if y != ey and x != ex:
    print(x, y)

ww = mt.atan2(y, x)
print(ww)
ww = ww*(180/np.pi)
print(ww)