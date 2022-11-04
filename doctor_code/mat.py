import numpy as np
#matrix = np.load('/home/sqy/acerrun_0/saved_trajfollow/true_iter1.npy')
#datax2 = np.load('/home/sqy/acerrun_0/saved_trajfollow/pred_iter1.npy')

from scipy import io
#mat1 = np.load('D:/pycharm/program/model44/run_0/saved_trajfollow/true_iter1.npy')
#mat2 = np.load('D:/pycharm/program/model44/run_0/saved_trajfollow/pred_iter1.npy')
#mat3 = mat2[0][:,0:3]
#print (mat3.shape,mat3[0],mat3[2])
#io.savemat('ture.mat',{'ture':mat1})
#io.savemat('pred.mat',{'pred':mat3})
###mat3 = np.squeeze(mat2,axis=1)  #去除数组某行为1的列
#####把npy文件转化为mat文件，把数据放到MTLAB中运行，输出是视频

datax = np.load('E:/pycharm2018/project/circle_0411zaixian/run_0/training_data/states_val.npy')          ###[20,200,8]
datax1 = np.load('E:/pycharm2018/project/circle_0411zaixian/run_0/saved_forwardsim/predicted_100step0.npy')  ###[101,2000,8]
print (datax.shape,datax1.shape)

datax = datax[16,20:120,:]
#x0 = datax[16,:,0]
#y0 = datax[16,:,1]
datax1 = datax1[20,1600:1700,:]




#x1 =
print (datax.shape,datax1.shape)
#np.save('/states_vals'+'.npy',datax)
#np.save('run_0'+'/predicted'+'.npy',datax1)
#io.savemat('states.mat',{'states':datax})
#io.savemat('predicted.mat',{'predicted':datax1})