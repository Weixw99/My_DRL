import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

#import pandas as pd
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

datax=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/true_iter0.npy')

data_offline = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter0.npy')
pred_iter1=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter1.npy')
pred_iter2=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter2.npy')
pred_iter3=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter3.npy')
pred_iter4=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter4.npy')
pred_iter5=np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/saved_trajfollow/pred_iter5.npy')
#pred_iter6=np.load('E:/pycharm2018/project/circle_0425zaixian/run_0/saved_trajfollow/pred_iter6.npy')


#datax=np.load('D:/pycharm/program/model10/run_31/saved_trajfollow/true_iter1.npy')
#datay=np.load('D:/pycharm/program/model10/run_31/saved_trajfollow/pred_iter6.npy')

datax_true=datax[m:len(data_offline[0])]

pred0=data_offline[:,m:len(data_offline[0]),:]
pred1=pred_iter1[:,m:len(pred_iter1[0]),:]
pred2=pred_iter2[:,m:len(pred_iter1[0]),:]
pred3=pred_iter3[:,m:len(pred_iter1[0]),:]
pred4=pred_iter4[:,m:len(pred_iter1[0]),:]
pred5=pred_iter5[:,m:len(pred_iter1[0]),:]
#pred6=pred_iter6[:,m:len(pred_iter1[0]),:]




#datax=datax[2850:2951]
#datax1=datax1[:,2850:2951,:]
#datax=datax[0:]
print (datax_true.shape,pred1.shape,len(pred_iter1[0]))
x0=datax_true[:,0]
y0=datax_true[:,1]

x_offline=pred0[0,:,0]
y_offline=pred0[0,:,1]

x1=pred1[0,:,0]
y1=pred1[0,:,1]

x2=pred2[0,:,0]
y2=pred2[0,:,1]

x3=pred3[0,:,0]
y3=pred3[0,:,1]

x4=pred4[0,:,0]
y4=pred4[0,:,1]

x5=pred5[0,:,0]
y5=pred5[0,:,1]

#x6=pred6[0,:,0]
#y6=pred6[0,:,1]

pred1_=np.arange(m,len(pred0[0,:,0]),1)

e_offline=np.sqrt(np.multiply((y0-y_offline),(y0-y_offline)))
#print(x0[0],x1[0],pred1_[0],pred1_.shape,e_offline.shape)
error_offline, = plt.plot(0.1*pred1_,e_offline,color="orange",linewidth=2)

#e_1=np.sqrt(np.multiply((x0-x1),(x0-x1)))
#print(x0[0],x1[0],pred1_[0],pred1_.shape,e_1.shape)
#error_1, = plt.plot(0.1*pred1_,e_1,color="red",linewidth=2)
#blue_line = mlines.Line2D([], [], color='blue',label='X_error')
#plt.legend(handles=[blue_line])
#A7, = plt.plot(x7,y7,color="brown",linewidth=3,linestyle='-')

#blue_line = mlines.Line2D([], [], color='blue',label='X_error')
#plt.legend(handles=[blue_line])
#A7, = plt.plot(x7,y7,color="brown",linewidth=3,linestyle='-')
#e_2=np.sqrt(np.multiply((x0-x2),(x0-x2)))
#print(x0[0],x1[0],pred1_[0],pred1_.shape,e_1.shape)
#error_2, = plt.plot(0.1*pred1_,e_2,color="brown",linewidth=2)

#e_3=y0-y_offline

#error_3, = plt.plot(0.1*pred1_,e_3,color="grey",linewidth=2)

#e_4=np.sqrt(np.multiply((x0-x4),(x0-x4)))

#error_4, = plt.plot(0.1*pred1_,e_4,color="green",linewidth=2)

#e_5=np.sqrt(np.multiply((x0-x5),(x0-x5)))

#error_5, = plt.plot(0.1*pred1_,e_5,color="blue",linewidth=2)


#e_6=np.sqrt(np.multiply((x0-x6),(x0-x6)))

#error_6, = plt.plot(0.1*pred1_,e_6,color="cyan",linewidth=2)



#plt.legend([error_offline, error_1,error_2,error_3], [ "pred0","pred1","pred2","pred3"])










"""


pred1_=np.arange(m,len(pred1[0,:,0]),1)
e=np.sqrt(np.multiply((x0-x1),(x0-x1)))
print(x0[0],x1[0],pred1_[0],pred1_.shape,e.shape)
error, = plt.plot(0.1*pred1_,e,color="blue",linewidth=1)
blue_line = mlines.Line2D([], [], color='blue',label='X_error')
plt.legend(handles=[blue_line])

"""
plt.show()
# 存为图像
# fig.savefig('test.png')

