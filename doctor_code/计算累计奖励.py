#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/20 0020 20:08
# @Author  : Chao Pan  
# @File    : 计算累计奖励.py


import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

import pandas as pd
import numpy as np

from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np

fig = plt.figure(1, (5, 5)) ####图的长宽

ax = SubplotZero(fig, 1, 1, 1)  ####分割fig几部分
fig.add_subplot(ax)

"""新建坐标轴"""
#ax.axis["xzero"].set_visible(True)
#ax.axis["xzero"].label.set_text("新建y=0坐标")
#ax.axis["xzero"].label.set_color('green')
#ax.axis['yzero'].set_visible(True)
#ax.axis["yzero"].label.set_text("新建x=0坐标")

# 新建一条y=2横坐标轴
#ax.axis["新建1"] = ax.new_floating_axis(nth_coord=0, value=-25,axis_direction="bottom")
#ax.axis["新建1"].toggle(all=True)
#ax.axis["新建1"].label.set_text("X/m")
#ax.axis["新建1"].label.set_color('black')

offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建2"] = new_axisline(loc="left", offset=offset, axes=ax)
ax.axis["新建2"].label.set_text("Y/m")
ax.axis["新建2"].label.set_color('black')

offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建3"] = new_axisline(loc="bottom", offset=offset, axes=ax)
ax.axis["新建3"].label.set_text("X/m")
ax.axis["新建3"].label.set_color('black')

"""坐标箭头"""
#    ax.axis["xzero"].set_axisline_style("-|>")

"""隐藏坐标轴"""
# 方法一：隐藏上边及右边
# ax.axis["right"].set_visible(False)
# ax.axis["top"].set_visible(False)
#方法二：可以一起写
#ax.axis["top",'right'].set_visible(False)
# 方法三：利用 for in
# for n in ["bottom", "top", "right"]:
#     ax.axis[n].set_visible(False)
#ax.axis["bottom",'left'].set_visible(False)
"""设置刻度"""
#ax.set_ylim(-25, 25)
#ax.set_yticks([-0.01,0,0.01])
##   ax.set_yticks([-0.5,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2])
#ax.set_xlim([-25, 25])
#ax.set_xticks([-5,5,1])

#设置网格样式
#ax.grid(True, linestyle='-.')


list_accumulate_reward_offline = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/list_accumulate_reward0.npy')
list_accumulate_reward_iter1=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/list_accumulate_reward1.npy')
list_accumulate_reward_iter2=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/list_accumulate_reward2.npy')
list_accumulate_reward_iter3=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/list_accumulate_reward3.npy')
#datax4=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter4.npy')
#datax5=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter5.npy')
#datax6=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter6.npy')
#datax7=np.load('E:/pycharm2018/project/circle_0411zaixian/run_0/saved_trajfollow/pred_iter7.npy')
#datax8=np.load('E:/pycharm2018/project/circle_0411zaixian/run_0/saved_trajfollow/pred_iter8.npy')

print("list_accumulate_reward_offline为：",list_accumulate_reward_offline)
print(list_accumulate_reward_offline[0],list_accumulate_reward_offline[4499])
print(list_accumulate_reward_offline.shape)  # (4500,)

print(len(list_accumulate_reward_offline))  # 4500

all_step = len(list_accumulate_reward_offline)

# 先写离线数据的累计奖励
gamma = 0.99  # 折扣系数
#R = list_accumulate_reward_offline[0]

# n次方
def power(x, n): #如def power (x,n=2) 设置了n的默认值为2，x的n次方
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s

sum_R = 0
for step in range(all_step):

    R = list_accumulate_reward_offline[step]
    #a = power(gamma, step)

    #sum_R = sum_R + a * R
    sum_R = sum_R +  R


print("sum_R:",sum_R)  # 应该是可以的,   3223.91908856,,gamma = 1时，3963.14264563









datax=datax[0:len(datax5[0])]
#datax=datax[0:97]
#datax1=datax1[:,0:97,:]
print (datax.shape,datax5.shape)
x0=datax[:,0]
y0=datax[:,1]
#fi1=datax1[:,2]
#x1=datax1[:,:,10]
#y1=datax1[:,:,11]
x_offline=data_offline[0,:,0]
y_offline=data_offline[0,:,1]


x1=datax1[0,:,0]
y1=datax1[0,:,1]
x2=datax2[0,:,0]
y2=datax2[0,:,1]
x3=datax3[0,:,0]
y3=datax3[0,:,1]
x4=datax4[0,:,0]
y4=datax4[0,:,1]
x5=datax5[0,:,0]
y5=datax5[0,:,1]
x6=datax6[0,:,0]
y6=datax6[0,:,1]
#x7=datax7[0,:,0]
#y7=datax7[0,:,1]
#x8=datax8[0,:,0]
#y8=datax8[0,:,1]
#print(x0.shape,y0.shape)
#plt.scatter(x0,y0,color="black",linewidth=2)
#plt.scatter(x1,y1,color="red",linewidth=0.2)
plt.figure(1)#创建图表1


D0, = plt.plot(x0,y0,color="black",linewidth=5,linestyle='--')

A_offline, = plt.plot(x_offline,y_offline,color="orange",linewidth=6,linestyle='-')
A1, = plt.plot(x1,y1,color="red",linewidth=4,linestyle='-')
A2, = plt.plot(x2,y2,color="brown",linewidth=3,linestyle='-')
A3, = plt.plot(x3,y3,color="grey",linewidth=3,linestyle='-')
A4, = plt.plot(x4,y4,color="green",linewidth=3,linestyle='-')

#plt.legend([D0, A4], [ "Desire","RL=4"])
#plt.legend([D0, A1,A2], [ "Desire","RL=1末","RL=2始"])
#plt.legend([D0, A1,A2,A3], [ "Desire","True1","True2","True3"])
#plt.legend([ D0,A1,A2,A3,A4], ["Desire","True1","Ture2","Ture3","Ture4" ])
plt.legend([ D0,A_offline,A1,A2,A3,A4], ["Desire","True0","True1","Ture2","Ture3","Ture4" ])
# 于 offset 处新建一条纵坐标
#  offset = (40, 0)
#  new_axisline = ax.get_grid_helper().new_fixed_axis
#  ax.axis["新建2"] = new_axisline(loc="right", offset=offset, axes=ax)
#  ax.axis["新建2"].label.set_text("新建纵坐标")
#  ax.axis["新建2"].label.set_color('red')
plt.figure(2)#创建图表2



#D0, = plt.plot(x0,y0,color="black",linewidth=5,linestyle='--')
#A5, = plt.plot(x5,y5,color="blue",linewidth=6,linestyle='-')
#A6, = plt.plot(x6,y6,color="cyan",linewidth=3,linestyle='-')
#A7, = plt.plot(x7,y7,color="brown",linewidth=3,linestyle='-')
#A8, = plt.plot(x8,y8,color="green",linewidth=3,linestyle='-')
#plt.legend([D0, A5,A6], [ "Desire","True5","True6"])
plt.show()
# 存为图像

#fig.savefig('Cumulative Reward.pdf')






