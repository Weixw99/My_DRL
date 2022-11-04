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
ax.axis["新建2"].label.set_text("||e||")
ax.axis["新建2"].label.set_color('black')

offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建3"] = new_axisline(loc="bottom", offset=offset, axes=ax)
ax.axis["新建3"].label.set_text("Time(s)")
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

###起始参数设置
m = 0

datax=np.load('D:/pycharm/program/model10/run_31/saved_trajfollow/true_iter1.npy')
datay=np.load('D:/pycharm/program/model10/run_31/saved_trajfollow/pred_iter6.npy')
datax1=datax[m:len(datay[0])]
datay1=datay[:,m:len(datay[0]),:]
#datax=datax[2850:2951]
#datax1=datax1[:,2850:2951,:]
#datax=datax[0:]
print (datax1.shape,datay1.shape,len(datay[0]))
x0=datax1[:,0]
y0=datax1[:,1]

x1=datay1[0,:,0]
y1=datay1[0,:,1]

x2=np.arange(m,len(datay[0,:,0]),1)
e=np.sqrt(np.multiply((x0-x1),(x0-x1)) + np.multiply((y0-y1),(y0-y1)))
print(x0[0],x1[0],x2[0],x2.shape,e.shape)
error, = plt.plot(0.1*x2,e,color="blue",linewidth=1)
blue_line = mlines.Line2D([], [], color='blue',label='Cross_error')
plt.legend(handles=[blue_line])
plt.show()
# 存为图像
# fig.savefig('test.png')

