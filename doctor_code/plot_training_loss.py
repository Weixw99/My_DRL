import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

import pandas as pd
import numpy as np

from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np

fig = plt.figure(1, (7, 5)) ####图的长宽

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

"""

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

"""
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



list_new_loss = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/losses/list_new_loss.npy')
list_old_loss = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/losses/list_old_loss.npy')
list_training_loss = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/losses/list_training_loss.npy')

print("list_new_loss为：",list_new_loss)  # [ 0.          0.01400006  0.01850381  0.02117461  0.02237916  0.02117025]
print("list_new_loss.shape为：",list_new_loss.shape)  # (9,)

print("list_old_loss为：",list_old_loss)  #
print("list_old_loss.shape为：",list_old_loss.shape)  # (9,)


print("list_training_loss为：",list_training_loss)  #
print("list_training_loss.shape为：",list_training_loss.shape)  # (9,)

print("list_training_loss.shape[0]为：",list_training_loss.shape[0])
list_training_loss_num  = list(range(list_training_loss.shape[0]))
print("list_training_loss_num为：",list_training_loss_num)  #  [0, 1, 2, 3, 4, 5, 6, 7, 8]

#plt.scatter(x0,y0,color="black",linewidth=2)
#plt.scatter(x1,y1,color="red",linewidth=0.2)
plt.figure(1)#创建图表1


training_loss, = plt.plot(list_training_loss_num,list_training_loss,color="red",linewidth=1)



#plt.legend([D0, A4], [ "Desire","RL=4"])
#plt.legend([D0, A1,A2], [ "Desire","RL=1末","RL=2始"])
#plt.legend([D0, A1,A2,A3], [ "Desire","True1","True2","True3"])
#plt.legend([ D0,A1,A2,A3,A4], ["Desire","True1","Ture2","Ture3","Ture4" ])

# 于 offset 处新建一条纵坐标
#  offset = (40, 0)
#  new_axisline = ax.get_grid_helper().new_fixed_axis
#  ax.axis["新建2"] = new_axisline(loc="right", offset=offset, axes=ax)
#  ax.axis["新建2"].label.set_text("新建纵坐标")
#  ax.axis["新建2"].label.set_color('red')
#plt.figure(2)#创建图表2


#plt.legend([D0, A6,A7,A8], [ "Desire","True6","True7","True8"])
plt.show()
# 存为图像
# fig.savefig('test.png')

