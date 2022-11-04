import time
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
ax.axis["新建2"].label.set_text("||e||")
ax.axis["新建2"].label.set_color('black')

offset = (0, 0)
new_axisline = ax.get_grid_helper().new_fixed_axis
ax.axis["新建3"] = new_axisline(loc="bottom", offset=offset, axes=ax)
ax.axis["新建3"].label.set_text("Time(s)")
ax.axis["新建3"].label.set_color('black')
#datax=np.load('/C:/sqy/modelss/run_0/training_data/dataX.npy')
run_num=6
num_aggregation_iters=
datax=np.load('D:/pycharm/program/model11/run_' +str(run_num)+ '/saved_trajfollow/true_iter1.npy')
datax1=np.load('D:/pycharm/program/model11/run_' +str(run_num)+ '/saved_trajfollow/pred_iter'+str(num_aggregation_iters)+'.npy')

starting = time.time()
#datax=np.load('/home/sqy/model4/run_0/saved_trajfollow/true_iter1.npy')
#datax1=np.load('/home/sqy/model4/run_0/saved_trajfollow/pred_iter1.npy')

#datax=datax[105:141]
#datax1=datax1[:,000:7000,:]
ax.set_ylim(-0.0, 0.4)

m=0
print(datax.shape)
datax=datax[m:len(datax1[0])]
datax1=datax1[:,m:len(datax1[0]),:]
print (datax1.shape)
#print(datax[5000],datax1[0][5000-1])
x0=datax[:,0]
y0=datax[:,1]
fi0=datax[:,2]
#fi1=datax1[:,2]
#x1=datax1[:,:,10]
#y1=datax1[:,:,11]
x1=datax1[:,:,0]
y1=datax1[:,:,1]
z1=datax1[:,:,2]
x=np.arange(0,len(z1[0]-1),1)
#xx=np.array([x])
#xxx=xx[:,x]
y=z1[0,x]
print(x.shape,fi0.shape)
#plt.scatter(x0,y0,color="black",linewidth=0.2)
#plt.plot(x1,y1,color="red",linewidth=0.2)
error = np.sqrt(np.multiply((y-fi0),(y-fi0)))
plt.plot(0.1*x,error,color="blue",linewidth=1)
blue_line = mlines.Line2D([], [], color='blue',label=r'$\psi$_error'+str(num_aggregation_iters))
plt.legend(handles=[blue_line])
plt.show()
#fig.savefig('test.png')
#print (datay[0],datay.shape)

