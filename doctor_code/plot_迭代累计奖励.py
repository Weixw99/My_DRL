
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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

"""

fig = plt.figure(1, (8, 5)) ####图的长宽
ax = SubplotZero(fig, 1, 1, 1)  ####分割fig几部分
fig.add_subplot(ax)
"""

fig, ax = plt.subplots(1, 1, figsize=(6, 4)) ####图的长宽

accumulate_reward0 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward0.npy')
accumulate_reward1 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward1.npy')
accumulate_reward2 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward2.npy')
accumulate_reward3 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward3.npy')
accumulate_reward4 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward4.npy')
accumulate_reward5 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward5.npy')
accumulate_reward6 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward6.npy')
#accumulate_reward7 = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/accumulate_reward7.npy')


#epoch_training_loss = epoch_training_loss.flatten()  # 降维，data = [[1],[2]] ---> data = data.flatten() ---> [1,2]

print("accumulate_reward0为：",accumulate_reward0)
print("accumulate_reward0.shape为：",accumulate_reward0.shape)
print("accumulate_reward1为：",accumulate_reward1)
print("accumulate_reward1.shape为：",accumulate_reward1.shape)
print("accumulate_reward2为：",accumulate_reward2)
print("accumulate_reward2.shape为：",accumulate_reward2.shape)
print("accumulate_reward3为：",accumulate_reward3)
print("accumulate_reward3.shape为：",accumulate_reward3.shape)
print("accumulate_reward4为：",accumulate_reward4)
print("accumulate_reward4.shape为：",accumulate_reward4.shape)
print("accumulate_reward5为：",accumulate_reward5)
print("accumulate_reward5.shape为：",accumulate_reward5.shape)
#print("accumulate_reward6为：",accumulate_reward6)
#print("accumulate_reward6.shape为：",accumulate_reward6.shape)
#print("accumulate_reward7为：",accumulate_reward7)
#print("accumulate_reward7.shape为：",accumulate_reward7.shape)



reward_array = np.array([-accumulate_reward0,-accumulate_reward1,-accumulate_reward2,
                         -accumulate_reward3,-accumulate_reward4,-accumulate_reward5])  # ,-accumulate_reward7



print("reward_array为：",reward_array)
iteration = np.array([0,1,2,3,4,5])  # ,7


#print("epoch_training_loss为：",epoch_training_loss)
#print("epoch_training_loss.shape为：",epoch_training_loss.shape)  # (60,)

#epoch_training_loss_num  = list(range(epoch_training_loss.shape[0]))
#print("epoch_training_loss_num为：",epoch_training_loss_num) # [0, 1, 2,....59]
#red,   = plt.plot(x1,y1,color="red",linewidth=2)

#plt.figure(1)#创建图表1


plot_Cumulative_Reward, = plt.plot(iteration ,reward_array,color="red",linewidth=2,marker=".",markersize=7)


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

plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Cumulative Reward')

#plt.title('This is Helvetica Font', fontweight='normal')

#
#plt.yticks(fontproperties = 'Arial', size = 18)
#plt.xticks(fontproperties = 'Arial', size = 18)

#plt.yticks(fontproperties = 'Arial', size = 18)
#plt.xticks(fontproperties = 'Arial', size = 18)

# 嵌入绘制局部放大图的坐标系
axins = inset_axes(ax, width="50%", height="30%",loc='upper left',
                   bbox_to_anchor=(0.4, -0.1, 1, 1),
                   bbox_transform=ax.transAxes)



axins.plot(iteration ,reward_array, color='r', linewidth=1,
            marker='o', markersize=5)

# 设置放大区间
zone_left = 3
zone_right = 5

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.4 # x轴显示范围的扩展比例
y_ratio = 0.8 # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = iteration[zone_left]-(iteration[zone_right]-iteration[zone_left])*x_ratio   # 2.07
xlim1 = iteration[zone_right]+(iteration[zone_right]-iteration[zone_left])*x_ratio


# Y轴的显示范围
y = np.hstack((reward_array[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

my_x1 = np.arange(3,6,1)

plt.xticks(my_x1)

plt.grid()
plt.show()

fig.savefig('Iteration cumulative reward.pdf')

plt.show()
