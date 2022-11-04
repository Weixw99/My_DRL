
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib
#print(matplotlib.matplotlib_fname())
#print(matplotlib.get_cachedir())

#matplotlib.rcParams['backend'] = 'SVG'
from mpl_toolkits.axisartist.axislines import SubplotZero

#plt.rcParams['axes.linewidth'] = 1  # 图框宽度
#plt.rcParams['figure.dpi'] = 100  # plt.show显示分辨率

# 坐标轴的刻度设置向内(in)或向外(out)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#plt.rcParams['font.weight'] = 'bold'   # bold加粗
#plt.rcParams["font.family"] = "Helvetica"


fig = plt.figure(1, (6, 5)) ####图的长宽
ax = SubplotZero(fig, 1, 1, 1)  ####分割fig几部分
fig.add_subplot(ax)


list_accumulate_reward1 = np.load('E:/pycharm2018/project/circle_0331/run_0/list_accumulate_reward1.npy')
#epoch_training_loss = epoch_training_loss.flatten()  # 降维，data = [[1],[2]] ---> data = data.flatten() ---> [1,2]

print("list_accumulate_reward1为：",list_accumulate_reward1)
print("list_accumulate_reward1.shape为：",list_accumulate_reward1.shape)  #  (4201,)

# 数组转列表
accumulate_reward1_2 = list_accumulate_reward1[1:4201]*-1

print("accumulate_reward1_2为：",accumulate_reward1_2)
print("accumulate_reward1_2.shape为：",accumulate_reward1_2.shape)

accumulate_reward1_2_num  = list(range(accumulate_reward1_2.shape[0]))
print("accumulate_reward1_2_num为：",accumulate_reward1_2_num) # [0, 1, 2,....4199]
accumulate_reward1_2_num = np.array(accumulate_reward1_2_num)

accumulate_reward1_2_time = accumulate_reward1_2_num*0.1
print("accumulate_reward1_2_time为：",accumulate_reward1_2_time)  # [0, 0.1, 0.2,....419.9]

epoch_training_loss_plot, = plt.plot(accumulate_reward1_2_time ,accumulate_reward1_2,color="red",linewidth=1)

#print("epoch_training_loss为：",epoch_training_loss)
#print("epoch_training_loss.shape为：",epoch_training_loss.shape)  # (60,)

#epoch_training_loss_num  = list(range(epoch_training_loss.shape[0]))
#print("epoch_training_loss_num为：",epoch_training_loss_num) # [0, 1, 2,....59]
#red,   = plt.plot(x1,y1,color="red",linewidth=2)

#plt.figure(1)#创建图表1


#epoch_training_loss_plot, = plt.plot(epoch_training_loss_num ,epoch_training_loss,color="red",linewidth=1,marker=".",markersize=7)


"""
plt.tick_params(labelsize=20)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Microsoft JhengHei') for label in labels]

"""



font = {'family': 'sans-serif',
        'sans-serif': 'Helvetica',
        'weight': 'black',
        'size': 15}   # normal
plt.rc('font', **font)  # pass in the font dict as kwargs



"""
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 9.5,
}

plt.rc('font', **font2)  # pass in the font dict as kwargs
"""

#plt.rc('font', **font)  # pass in the font dict as kwargs

#plt.xlabel('Epoch',fontweight='normal')
#plt.ylabel('Loss',fontweight='normal')

plt.xlabel('Time(s)')
plt.ylabel('Reward')


# 嵌入绘制局部放大图的坐标系
axins = inset_axes(ax, width="90%", height="30%",loc='lower left',
                   bbox_to_anchor=(0.25, 0.1, 0.7, 0.7),
                   bbox_transform=ax.transAxes)



axins.plot(accumulate_reward1_2_time ,accumulate_reward1_2, color='b', linewidth=1)

# 设置放大区间
zone_left = 1000
zone_right = 4100

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5 # x轴显示范围的扩展比例
y_ratio = 0.25   # y轴显示范围的扩展比例

print("accumulate_reward1_2_time[zone_left]为:",accumulate_reward1_2_time[zone_left])  # 0.1

# X轴的显示范围
#xlim0 = accumulate_reward1_2_time[zone_left]-(accumulate_reward1_2_time[zone_right]-accumulate_reward1_2_time[zone_left])*x_ratio   # 2.07
#xlim1 = accumulate_reward1_2_time[zone_right]+(accumulate_reward1_2_time[zone_right]-accumulate_reward1_2_time[zone_left])*x_ratio
xlim0 = accumulate_reward1_2_time[zone_left]
xlim1 = accumulate_reward1_2_time[zone_right]

# Y轴的显示范围
y = np.hstack((accumulate_reward1_2[zone_left:zone_right]))
print("y为：",y)
print("y.shape为：",y.shape)

ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio


# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

my_x1 = np.arange(100,410,50)

plt.xticks(my_x1)







#ax = plt.axes([0.2,0.55,0.3,0.3]) # 参数含义[left, bottom, width, height]
#ax.plot(accumulate_reward1_2_time ,accumulate_reward1_2,color = 'g')
#plt.title('This is Helvetica Font', fontweight='normal')

#
#plt.yticks(fontproperties = 'Arial', size = 18)
#plt.xticks(fontproperties = 'Arial', size = 18)

#plt.yticks(fontproperties = 'Arial', size = 18)
#plt.xticks(fontproperties = 'Arial', size = 18)

#plt.savefig('figure.eps')
plt.show()
#fig.savefig('Loss.svg',format='svg')
#fig.savefig('Cumulative_Reward.pdf')


#plt.savefig('Cumulative_Reward.pdf')
