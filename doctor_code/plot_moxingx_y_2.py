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
run_num = 0

"""


datax=np.load('E:/pycharm2018/project/model11/run_'  +str(run_num)+ '/saved_trajfollow/true_iter1.npy')
datax1=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter1.npy')
datax2=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter2.npy')
datax3=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter3.npy')
datax4=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter4.npy')
datax5=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter5.npy')
datax6=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter6.npy')
datax7=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter7.npy')
datax8=np.load('E:/pycharm2018/project/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter8.npy')
#datax9=np.load('D:/pycharm/program/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter9.npy')

"""



states_val = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/training_data/states_val.npy')          ###[5,200,8]
predicted_100step0 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_forwardsim/predicted_100step0.npy')  ###[101,2000,8]


#pred100_5 = predicted_100step5[5]
#(5, 200, 8) (101, 500, 8)
print(states_val.shape,predicted_100step0.shape)


s_val = states_val[3,1:61,:]

print("s_val为：",s_val)  # [0.2,0.3,,,,,,,,,,,,,]
print("s_val.shape为：",s_val.shape)  # (100,8)

#pred1 = predicted_100step0[1,1:101,:]
pred1 = predicted_100step0[1,300:360,:]
#pred2 = predicted_100step0[1,300:340,:]
#pred3 = predicted_100step0[1,300:360,:]
#pred4 = predicted_100step0[1,303:403,:]


pred30 = predicted_100step0[30,300:360,:]


# 这种是对应不同起始点
#pred2 = predicted_100step0[2,300:400,:]
#pred3 = predicted_100step0[3,300:400,:]
#pred4 = predicted_100step0[4,300:400,:]
#pred5 = predicted_100step0[5,300:400,:]
#pred90 = predicted_100step0[90,300:400,:]


#pred2 = predicted_100step0[1,300:350,:]
#pred1 = predicted_100step0[1,300:400,:]
#pred3 = predicted_100step0[3,20:120,:]



print("pred1[0]为：",pred1[0])  # [0.2,0.3,,,,,,,,,,,,,]
print("pred1.shape为：",pred1.shape)  # (100,8)
print("pred2[0]为：",pred2[0])  # [0.2,0.3,,,,,,,,,,,,,]
print("pred2.shape为：",pred2.shape)  # (100,8)
#print("pred3[0]为：",pred3[0])  # [0.2,0.3,,,,,,,,,,,,,]
#print("pred3.shape为：",pred3.shape)  # (100,8)

print("pred30[0]为：",pred30[0])  # [0.2,0.3,,,,,,,,,,,,,]
print("pred30.shape为：",pred30.shape)  # (100,8)

val_x = s_val[:,0]
val_y = s_val[:,1]

lab, = plt.plot(val_x,val_y,color="black",linewidth=3,linestyle='--')

pred_x1 = pred1[:,0]
pred_y1 = pred1[:,1]

pred_x2 = pred2[:,0]
pred_y2 = pred2[:,1]

#pred_x3 = pred3[:,0]
#pred_y3 = pred3[:,1]

#pred_x4 = pred4[:,0]
#pred_y4 = pred4[:,1]

#pred_x5 = pred5[:,0]
#pred_y5 = pred5[:,1]
#pred_x90 = pred90[:,0]
#pred_y90 = pred90[:,1]

#pred01, = plt.plot(pred_x1,pred_y1,color="red",linewidth=2,linestyle='--')

#pred_x1 = pred1[:,0]
#pred_y1 = pred1[:,1]

#pred_x2 = pred2[:,0]
#pred_y2 = pred2[:,1]

pred01, = plt.plot(pred_x1,pred_y1,color="red",linewidth=1,linestyle='--')

pred02, = plt.plot(pred_x2,pred_y2,color="blue",linewidth=2,linestyle='--')

#pred03, = plt.plot(pred_x3,pred_y3,color="orange",linewidth=2,linestyle='--')

#pred02, = plt.plot(pred_x2,pred_y2,color="green",linewidth=2,linestyle='--')
#pred090, = plt.plot(pred_x90,pred_y90,color="blue",linewidth=2,linestyle='--')

"""

states_val_0 = states_val[16,10:110,:]
#x0 = datax[16,:,0]
#y0 = datax[16,:,1]
predicted_100step0_0 = predicted_100step0[10,1600:1700,:]

# (100, 8) (100, 8)
print (states_val_0.shape,predicted_100step0_0.shape)



val_x = states_val_0[:,0]
val_y = states_val_0[:,1]

print("val_x为：",val_x)
print("val_x.shape为：",val_x.shape)


predicted_100step0_0_x = predicted_100step0_0[:,0]
predicted_100step0_0_y = predicted_100step0_0[:,1]

#print("labels_1step1的维度分别为：",labels_5step0_x)

val, = plt.plot(val_x,val_y,color="black",linewidth=1,linestyle='--')
step_100, = plt.plot(predicted_100step0_0_x,predicted_100step0_0_y,color="red",linewidth=1)

#x, = plt.plot(val_x,predicted_100step0_0_x,color="black",linewidth=1,linestyle='--')

#y, = plt.plot(val_y,predicted_100step0_0_y,color="red",linewidth=1)

#labels_1, = plt.plot(labels_5step0_x,labels_5step0_y,color="black",linewidth=5,linestyle='--')
#pred_100step0 = plt.plot(predicted_100step0_x,predicted_100step0_y,color="red",linewidth=3)


#plt.legend([D0, A6,A7,A8], [ "Desire","True6","True7","True8"])
plt.show()
# 存为图像
# fig.savefig('test.png')

"""

plt.show()

