#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/24 0024 21:40
# @Author  : Chao Pan  
# @File    : plot_muxingx.py

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
#plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

import pandas as pd
import numpy as np

from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np

fig = plt.figure(1, (5, 5)) ####图的长宽

ax = SubplotZero(fig, 1, 1, 1)  ####分割fig几部分
fig.add_subplot(ax)

states_val = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/training_data/states_val.npy')          ###[5,200,8]
predicted_100step3 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_forwardsim/predicted_100step3.npy')  ###[101,2000,8]

#pred100_5 = predicted_100step5[5]
# (5, 200, 8) (101, 500, 8)
print(states_val.shape,predicted_100step3.shape)

s_val = states_val[3,1:101,:]

print("s_val为：",s_val)  # [0.2,0.3,,,,,,,,,,,,,]
print("s_val.shape为：",s_val.shape)  # (100,8)

pred1 = predicted_100step3[1,300:400,:]
print("pred1为：",pred1)
#pred2 = predicted_100step0[1,301:351,:]

#pred2 = predicted_100step0[1,300:350,:]
#pred1 = predicted_100step0[1,300:400,:]
#pred3 = predicted_100step0[3,20:120,:]

#print("pred为：",pred1)  # [0.2,0.3,,,,,,,,,,,,,]
#print("pred.shape为：",pred1.shape)  # (100,8)

val_x = s_val[:,0]
print("len(val_x) 为：",len(val_x) )

#val_y = s_val[:,1]
val_x = s_val[:,0]

#print ("len(z1[0]-1)为：",len(z1[0]-1))
#pred_x1 = pred1[:,0]
#pred_y1 = pred1[:,1]
pred_x1 = pred1[:,0]

#z2=datax1[:,:,3]
x=np.arange(0,len(val_x),1)
#y=z2[0,x]

lab, = plt.plot(x*0.1,val_x,color="black",linewidth=1)

#pred_x2 = pred2[:,0]
#pred_y2 = pred2[:,1]

pred01, = plt.plot(x*0.1,pred_x1,color="red",linewidth=2,linestyle='--')
#pred02, = plt.plot(pred_x2,pred_y2,color="blue",linewidth=2,linestyle='--')

plt.show()
# 存为图像
# fig.savefig('test.png')

