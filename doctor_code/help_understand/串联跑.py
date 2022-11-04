

import numpy as np

curr_state = np.copy(forwardsim_x_true[0])     # curr state is of dim NN input,,,

for curr_control in forwardsim_y:  ####数组forwardsim_y[4660,100,2],这种 for的取法是挨个取4660个二维数组,(100,2)

    state_list.append(np.copy(curr_state))

    ##取第一维数为变量，逐个取到forwardsim_y，先取第1个（100，2）2维数组，
    # 然后通过np.expand_dims（curr_control, axis=0）将其变成3维数组
    # 现在curr_control为3维数组,(100,2)，是将forwardsim_y里面的每一个2维作为整体，输出时变为3维
    # (1,100,2)
    curr_control = np.expand_dims(curr_control, axis=0)

    # subtract mean and divide by standard deviation 减去平均数，再除以标准差
    # self.mean_x和self.std_x是（1,16）
    # 下面两行是0均值标准化，经过处理的数据符合标准正态分布，即均值为0，标准差为1
    curr_state_preprocessed = curr_state - self.mean_x
    # 现在的curr_state_preprocessed是1维的，（1,16）
    curr_state_preprocessed = np.nan_to_num(curr_state_preprocesse d /self.std_x)


    # self.mean_y是1维的，(1,2)，而curr_control是3维的，发生python广播机制，进行相减，
    curr_control_preprocessed = curr_control - self.mean_y
    # 输出的curr_control_preprocessed是3维的，（1,100,2）
    curr_control_preprocessed = np.nan_to_num(curr_control_preprocesse d /self.std_y)


    # inputs_preprocessed形成(1,216 )  2维数组[[,,,,,]],在help_understand
    inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

    # run through NN to get prediction,,,,输出的model_output维度也是(1,216),2维的？？？
    # curr_nn_output是前馈神经网络输出的结果，具体输出的数据维度？？
    model_output = sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed})

    # 反归一化，形成正常值
    # multiply by std and add mean back in 乘以标准差，并加上均值
    # model_output若是2维，model_output[0][0]指的是里面具体的元素，是一个数字
    state_difference s= (model_output[0][0 ] *self.std_z ) +self.mean_z

    # update the state info
    next_state = curr_state + state_differences

    # copy the state info
    curr_stat e= np.copy(next_state)



