import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np

fig = plt.figure(1, (5, 5)) ####图的长宽

ax = SubplotZero(fig, 1, 1, 1)  ####分割fig几部分
fig.add_subplot(ax)
offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建2"] = new_axisline(loc="left", offset=offset, axes=ax)
ax.axis["新建2"].label.set_text("[N]")
ax.axis["新建2"].label.set_color('black')

offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建3"] = new_axisline(loc="bottom", offset=offset, axes=ax)
ax.axis["新建3"].label.set_text("Time(s)")
ax.axis["新建3"].label.set_color('black')
num_aggregation_iters=8
datax=np.load('D:/pycharm/program/model11/run_5/saved_trajfollow/control_iter'+str(num_aggregation_iters)+'.npy')
ax.set_ylim(-0.5, 2.5)
#ax.set_yticks([-0.01,0,0.01])
##   ax.set_yticks([-0.5,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2])
#ax.set_xlim([-5, 350])
#datax=datax[0:6000]
#datax1=datax1[:,0:20000,:]
x1=datax[:,:,0]
x=np.arange(0,len(x1-1),2)
y=x1[x,0]
#plt.scatter(x0,y0,color="black",linewidth=0.2)
#plt.scatter(x1,y1,color="red",linewidth=0.2)   marker='*',markersize=0,
plt.plot(0.1*x,y,color="blue",linewidth=1)
blue_line = mlines.Line2D([], [], color='blue',label=r'$\tau_u$')
plt.legend(handles=[blue_line])
#plt.savefig('fix.png', dpi=300)保存指定分辨率的图片
plt.show()


