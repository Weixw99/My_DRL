import time
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
offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建2"] = new_axisline(loc="left", offset=offset, axes=ax)
ax.axis["新建2"].label.set_text("[m/s]")
ax.axis["新建2"].label.set_color('black')

offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建3"] = new_axisline(loc="bottom", offset=offset, axes=ax)
ax.axis["新建3"].label.set_text("Time(s)")
ax.axis["新建3"].label.set_color('black')
#datax=np.load('/C:/sqy/modelss/run_0/training_data/dataX.npy')
datax=np.load('D:/pycharm/program/model11/run_6/saved_trajfollow/true_iter1.npy')
datax1=np.load('D:/pycharm/program/model11/run_6/saved_trajfollow/pred_iter6.npy')
#datax3=np.load('/home/sqy/nn_dynamics/run_1/training_data/forwardsim_y.npy')
#datax4=np.load('/home/sqy/nn_dynamics/run_1/training_data/forwardsim_x_true.npy')
#test=test.tolist()
#datay=np.load('/home/sqy/models/run_0/training_data/dataY.npy')
#test2=test2.tolist()
#dataz=np.load('/home/sqy/nn_dynamics/run_11/data_genatesss/dataZ.npy')
#x=np.array([1950,1970,1990,2010])
#y=np.array([2.4,3.86,5.31,8.64])
#datay=np.concatenate((datay),axis=1)
#datax1=datax1[0:800]
#datax=datax[0:120]
starting = time.time()
#datax=np.load('/home/sqy/model4/run_0/saved_trajfollow/true_iter1.npy')
#datax1=np.load('/home/sqy/model4/run_0/saved_trajfollow/pred_iter1.npy')

#datax=datax[105:141]
#datax1=datax1[:,000:7000,:]

#datax=datax[0:6000]
datax1=datax1[:,0:20000,:]
print (datax1.shape)
x0=datax[:,0]
y0=datax[:,1]
#fi1=datax1[:,2]
#x1=datax1[:,:,10]
#y1=datax1[:,:,11]
x1=datax1[:,:,0]
y1=datax1[:,:,1]
z1=datax1[:,:,2]
z2=datax1[:,:,3]
x=np.arange(0,len(z1[0]-1),1)
y=z2[0,x]
#plt.scatter(x0,y0,color="black",linewidth=0.2)
#plt.scatter(x1,y1,color="red",linewidth=0.2)   marker='*',markersize=0,
plt.plot(0.1*x,y,color="blue",linewidth=1)
blue_line = mlines.Line2D([], [], color='blue',label='u')
plt.legend(handles=[blue_line])
#plt.savefig('fix.png', dpi=300)保存指定分辨率的图片
#plt.scatter(x,yy,color="red",linewidth=0.2)
plt.show()
#print (datay[0],datay.shape)

