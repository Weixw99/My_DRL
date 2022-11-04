
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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


fig = plt.figure(1, (8, 5)) ####图的长宽
ax = SubplotZero(fig, 1, 1, 1)  ####分割fig几部分
fig.add_subplot(ax)


epoch_training_loss = np.load('E:/pycharm2018/project/circle_0427zaixian/run_0/all_loss.npy')
epoch_training_loss = epoch_training_loss.flatten()  # 降维，data = [[1],[2]] ---> data = data.flatten() ---> [1,2]

print("epoch_training_loss为：",epoch_training_loss)  # [ 0.          0.01400006  0.01850381  0.02117461  0.02237916  0.02117025]

#print("epoch_training_loss为：",epoch_training_loss)
print("epoch_training_loss.shape为：",epoch_training_loss.shape)  # (60,)

epoch_training_loss_num  = list(range(epoch_training_loss.shape[0]))
print("epoch_training_loss_num为：",epoch_training_loss_num) # [0, 1, 2,....59]
#red,   = plt.plot(x1,y1,color="red",linewidth=2)

#plt.figure(1)#创建图表1


epoch_training_loss_plot, = plt.plot(epoch_training_loss_num ,epoch_training_loss,color="red",linewidth=1,marker=".",markersize=7)


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

plt.xlabel('Epoch')
plt.ylabel('Loss')

#plt.title('This is Helvetica Font', fontweight='normal')

#
#plt.yticks(fontproperties = 'Arial', size = 18)
#plt.xticks(fontproperties = 'Arial', size = 18)

#plt.yticks(fontproperties = 'Arial', size = 18)
#plt.xticks(fontproperties = 'Arial', size = 18)

#plt.savefig('figure.eps')
plt.show()
#fig.savefig('Loss.svg',format='svg')
fig.savefig('Loss.pdf')


#plt.savefig('Loss.pdf')
