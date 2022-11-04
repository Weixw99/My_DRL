import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

import pandas as pd
import numpy as np

from mpl_toolkits.axisartist.axislines import SubplotZero

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)


fig = plt.figure(1, (5, 5)) ####图的长宽

ax = SubplotZero(fig, 1, 1, 1)  ####分割fig几部分
fig.add_subplot(ax)

# 运行numpy 的时候发现 数据显示不全，而是用省略号省略了大量数据
# 加上这句，去掉省略号
#np.set_printoptions(threshold=np.inf)

#np.set_printoptions(threshold=np.inf)

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

controls_val = np.load('E:/pycharm2018/project/model11/run_0/training_data/controls_val.npy')
dataX = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataX.npy')
dataX_new_iter0 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataX_new_iter0.npy')
dataX_new_iter1 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataX_new_iter1.npy')
dataX_new_iter2 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataX_new_iter2.npy')
dataX_new_iter3 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataX_new_iter3.npy')


dataY = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataY.npy')
dataY_new_iter0 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataY_new_iter0.npy')
dataY_new_iter1 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataY_new_iter1.npy')
dataY_new_iter2 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataY_new_iter2.npy')
dataY_new_iter3 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataY_new_iter3.npy')


dataZ = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataZ.npy')
dataZ_new_iter0 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataZ_new_iter0.npy')
dataZ_new_iter1 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataZ_new_iter1.npy')
dataZ_new_iter2 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataZ_new_iter2.npy')
dataZ_new_iter3 = np.load('E:/pycharm2018/project/model11/run_0/training_data/dataZ_new_iter3.npy')

#forwardsim_x_true = np.load('E:/pycharm2018/project/model11/run_0/training_data/forwardsim_x_true.npy')
#forwardsim_y  = np.load('E:/pycharm2018/project/model11/run_0/training_data/forwardsim_y.npy')

states_val  = np.load('E:/pycharm2018/project/model11/run_0/training_data/states_val.npy')



print("controls_val为：",controls_val)  #
print("controls_val.shape为：",controls_val.shape)  # (20,333,2)

# 验证X
print("dataX.shape为：",dataX.shape)  # (8300,16)
print("dataX_new_iter0.shape为：",dataX_new_iter0.shape)  # (0,16),就是空的
print("dataX_new_iter0为：",dataX_new_iter0)  # []
print("dataX_new_iter1.shape为：",dataX_new_iter1.shape)  # (2997,16)
print("dataX_new_iter2.shape为：",dataX_new_iter2.shape)  # (5994,16)

print('**************************************************')
print('*****************************************************')
# 验证Y
print("dataY.shape为：",dataY.shape)  # (8300,2)
print("dataY_new_iter0.shape为：",dataY_new_iter0.shape)  # (0,2),就是空的
print("dataY_new_iter0为：",dataY_new_iter0)  # []
print("dataY_new_iter1.shape为：",dataY_new_iter1.shape)  # (2997,2)
print("dataY_new_iter2.shape为：",dataY_new_iter2.shape)  # (5994,2)

print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
# 验证Z
print("dataZ.shape为：",dataZ.shape)  # (8300,16)
print("dataZ_new_iter0.shape为：",dataZ_new_iter0.shape)  # (0,16),就是空的
print("dataZ_new_iter0为：",dataZ_new_iter0)  # []
print("dataZ_new_iter1.shape为：",dataZ_new_iter1.shape)  # (2997,16)
print("dataZ_new_iter2.shape为：",dataZ_new_iter2.shape)  # (5994,16)

#print("forwardsim_x_true.shape为：",forwardsim_x_true.shape)  # (99, 16)
#print("forwardsim_y.shape为：",forwardsim_y.shape)  # (99, 2)


print("states_val.shape为：",states_val.shape)

"""

datax=np.load('E:/pycharm2018/project/circle_0326/run_0/saved_trajfollow/true_iter1.npy')
datax1=np.load('E:/pycharm2018/project/circle_0326/run_0/saved_trajfollow/pred_iter1.npy')

# 查看一下参考和预测的起始点
print("参考datax[0]为：",datax[0])  # 参考起始点为（20,0）
print("预测datax1为：",datax1)  # 预测起始点为（15,-1）

print("datax1.shape为：",datax1.shape)
datax=datax[0:len(datax1[0])]
#datax=datax[0:97]
#datax1=datax1[:,0:97,:]

x0=datax[:,0]
y0=datax[:,1]
#fi1=datax1[:,2]
#x1=datax1[:,:,10]
#y1=datax1[:,:,11]
x1=datax1[0,:,0]
y1=datax1[0,:,1]
#print(x0.shape,y0.shape)
#plt.scatter(x0,y0,color="black",linewidth=2)
#plt.scatter(x1,y1,color="red",linewidth=0.2)

#black, = plt.plot(x0,y0,color="black",linewidth=1, marker='d',ms=4,linestyle='--')  # 参考轨迹用”*“标记

black, = plt.plot(x0,y0,color="black",linewidth=4)
red,   = plt.plot(x1,y1,color="red",linewidth=2)


#plt.legend([red, black], ["True", "Desire"])



# 于 offset 处新建一条纵坐标
#  offset = (40, 0)
#  new_axisline = ax.get_grid_helper().new_fixed_axis
#  ax.axis["新建2"] = new_axisline(loc="right", offset=offset, axes=ax)
#  ax.axis["新建2"].label.set_text("新建纵坐标")
#  ax.axis["新建2"].label.set_color('red')


plt.show()
# 存为图像
#fig.savefig('单船轨迹跟踪.png')

"""