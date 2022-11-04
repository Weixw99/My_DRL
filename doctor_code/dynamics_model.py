
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math
#from usvtest import Vehicletest
from feedforward_network import feedforward_network


class Dyn_Model:

    def __init__(self, inputSize, outputSize, sess, learning_rate, batchsize, x_index, y_index, 
                num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal):
#    def __init__(self, inputSize, outputSize, sess, learning_rate, batchsize,  x_index, y_index,
#                num_fc_layers, depth_fc_layers, tf_datatype, print_minimal):

        #init vars
        self.sess = sess
        self.batchsize = batchsize   # nerve in one layer,must be 2^n
#       self.which_agent = which_agent
        self.x_index = x_index       # x_postion
        self.y_index = y_index       # y_postion
        self.inputSize = inputSize   # states+controls =6+2 = 8
        self.outputSize = outputSize # next_states-front_states = 6
        self.mean_x = mean_x         # mean_dataX (states)
        self.mean_y = mean_y         # mean_dataY (controls)
        self.mean_z = mean_z         # mean_dataZ,(states difference_value)
        self.std_x = std_x           # x_standard deviation
        self.std_y = std_y           # y_standard deviation
        self.std_z = std_z           # z_standard deviation
        self.print_minimal = print_minimal

        ####training data input NN
        self.x_ = tf.placeholder(tf_datatype, shape=[None, self.inputSize], name='x') # num = 6+2 = 8
        self.z_ = tf.placeholder(tf_datatype, shape=[None, self.outputSize], name='z') #num = 6


        ##NN output
        self.curr_nn_output = feedforward_network(self.x_, self.inputSize, self.outputSize, 
                                                num_fc_layers, depth_fc_layers, tf_datatype)
        ##USV output
#        self.out = Vehicletest(self.x_)

        #loss (mean of difference values )
        self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))


        # Compute gradients and update parameters
        self.opt = tf.train.AdamOptimizer(learning_rate) ## optimise
#        self.opt = tf.train.GradientDescentOptimizer(learning_rate)
        self.theta = tf.trainable_variables()            ## which paramters colude be trained

        ####grandient, variable
        self.gv = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta)
                    if g is not None]

        self.train_step = self.opt.apply_gradients(self.gv)

        
        #####################################################################
        #############training NN defined#####################################
        #####################################################################
    def train(self, dataX, dataZ, dataX_new, dataZ_new, nEpoch, save_dir, fraction_use_new): ###new:old=fraction_use_new*10:1

        # 在main里面，对应562行调用，dataX=inputs为（10000,8），dataZ=outputs为（10000,8）
        # dataX_new=inputs_new为（未知，8），dataZ_new=outputs_new为（未知,8）

        #init vars
        start = time.time()
        training_loss_list = []
        # 返回一维数组[0, 1, 2.....,9999]
        range_of_indeces = np.arange(dataX.shape[0])
        # 10000
        nData_old = dataX.shape[0]
        # 行数目
        num_new_pts = dataX_new.shape[0]

        saver = tf.train.Saver(max_to_keep=0)
#        if(counter_agg_iters>0):
#            restore_path = save_dir + '/models/model_aggIter' + str(counter_agg_iters) + '.ckpt'
#            saver.restore(sess, restore_path)
 #           self.curr_nn_output =
        #how much of new data to use per batch
###########################################################
        if(num_new_pts<(self.batchsize*fraction_use_new)):
            batchsize_new_pts = num_new_pts #use all of the new ones
        else:
            batchsize_new_pts = int(self.batchsize*fraction_use_new)

        ###################################
        #################
        ###################################

        #how much of old data to use per batch
        batchsize_old_pts = int(self.batchsize - batchsize_new_pts)   # 第0次迭代时，batchsize_new_pts=0，第一次迭代时，batchsize_old_pts=0

        # 这个是后写的，
        all_loss = []
        #training loop
        for i in range(nEpoch): # 60
            
            #reset to 0
            avg_loss=0
            num_batches=0

            #randomly order indeces (equivalent to shuffling dataX and dataZ)
            # old_indeces是一个1维数组，将0,1....9999所有的数无重复的进行随机排列输出一个数组，比如[8,4,67,0,900,30,24,.....]
            # old_indeces对应抽取的状态的位置，旧的索引
            old_indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)


            #train from both old and new dataset
            if(batchsize_old_pts>0): 

                #get through the full old dataset
                for batch in range(int(math.floor(nData_old / batchsize_old_pts))):


                    #randomly sample points from new dataset
                    if(num_new_pts==0):             ####num_new_pts=dataX_new.shape[0]   ##pts=points
                        dataX_new_batch = dataX_new
                        dataZ_new_batch = dataZ_new
                    else:
                        new_indeces = npr.randint(0,dataX_new.shape[0], (batchsize_new_pts,))
                        dataX_new_batch = dataX_new[new_indeces, :]
                        dataZ_new_batch = dataZ_new[new_indeces, :]

                    #walk through the randomly reordered "old data"
                    dataX_old_batch = dataX[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
                    dataZ_old_batch = dataZ[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]

    
                    #combine the old and new data
                    dataX_batch = np.concatenate((dataX_old_batch, dataX_new_batch))
                    dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))
                    ####  dataX_old_batch = nData_old / (batchsize_old_pts = int(self.batchsize - batchsize_new_pts)


                    #one iteration of feedforward training
                    _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
                                                                feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
                    training_loss_list.append(loss)
                    avg_loss+= loss
                    num_batches+=1

            #train completely from new set
            else:                     ##batchsize_old_pts=0
                for batch in range(int(math.floor(num_new_pts / batchsize_new_pts))):  #

                    #walk through the shuffled new data
                    dataX_batch = dataX_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]
                    dataZ_batch = dataZ_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]

                    #one iteration of feedforward training
####>????????????????????????????????????????????####
                    _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
                                                                feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})

                    training_loss_list.append(loss)
                    avg_loss+= loss
                    num_batches+=1

                #shuffle new dataset after an epoch (if training only on it)
                p = npr.permutation(dataX_new.shape[0])
                dataX_new = dataX_new[p]
                dataZ_new = dataZ_new[p]

            #save losses after an epoch
            np.save(save_dir + '/training_losses.npy', training_loss_list)
            if(not(self.print_minimal)):
                if((i%10)==0):
                    print("\n=== Epoch {} ===".format(i))
                    print ("loss: ", avg_loss/num_batches)
                    print("num_batches: ", num_batches)

            # 每一步的loss
            print("loss: ", avg_loss / num_batches)
            loss_list = [avg_loss / num_batches]
            all_loss.append(loss_list)  # 每一个epoch都放到all_loss

        all_loss = np.array(all_loss)
        print("all_loss.shape:",all_loss.shape)  # （60,1）
        #np.save(save_dir + '/all_loss.npy', all_loss)



        if(not(self.print_minimal)):
            #print ("Training set size: ", (nData_old + dataX_new.shape[0])) ##when the aggregation = 0,  it prints 8300.
            print("Training set size: ", ((batchsize_old_pts*num_batches) + dataX_new.shape[0]))  ##when the aggregation = 0,  it prints 8300.
            print("Training duration: {:0.2f} s".format(time.time()-start))

        #get loss of curr model on old dataset
        avg_old_loss=0
        iters_in_batch=0
        for batch in range(int(math.floor(nData_old / self.batchsize))):
            # Batch the training data
            dataX_batch = dataX[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = dataZ[batch*self.batchsize:(batch+1)*self.batchsize, :]
            #one iteration of feedforward training
            loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
            avg_old_loss+= loss
            iters_in_batch+=1
        old_loss =  avg_old_loss/iters_in_batch

        #get loss of curr model on new dataset
        avg_new_loss=0
        iters_in_batch=0
        for batch in range(int(math.floor(dataX_new.shape[0] / self.batchsize))):
            # Batch the training data
            dataX_batch = dataX_new[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = dataZ_new[batch*self.batchsize:(batch+1)*self.batchsize, :]
            #one iteration of feedforward training
            loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
            avg_new_loss+= loss
            iters_in_batch+=1
        if(iters_in_batch==0):
            new_loss=0
        else:
            new_loss =  avg_new_loss/iters_in_batch

        #done
        return all_loss,(avg_loss/num_batches), old_loss, new_loss

    def run_validation(self, inputs, outputs):

        #init vars
        nData = inputs.shape[0]
        avg_loss=0
        iters_in_batch=0

        for batch in range(int(math.floor(nData / self.batchsize))):
            # Batch the training data
            dataX_batch = inputs[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = outputs[batch*self.batchsize:(batch+1)*self.batchsize, :]

            #one iteration of feedforward training
            z_predictions, loss = self.sess.run([self.curr_nn_output, self.mse_], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})

            avg_loss+= loss
            iters_in_batch+=1

        #avg loss + all predictions
        print ("Validation set size: ", nData)
        print ("Validation set's total loss: ", avg_loss/iters_in_batch)

        return (avg_loss/iters_in_batch)


    #multistep prediction using the learned dynamics model at each step
    def do_forward_sim(self, forwardsim_x_true, forwardsim_y, many_in_parallel):

        # forwardsim_y为(3000, 17, 2)，3维
        # forwardsim_x_true为列表，

        #init vars
        state_list = []

        print("len(forwardsim_x_true):",len(forwardsim_x_true))   # 2
        if(many_in_parallel):
            #init vars

            print("forwardsim_y.shape为：",forwardsim_y.shape)
            N= forwardsim_y.shape[0]  # 3000
            print("N为：",N)
            horizon = forwardsim_y.shape[1] # 17
            print("horizon为：", horizon)
#            print (N,horizon)
            array_stdz = np.tile(np.expand_dims(self.std_z, axis=0),(N,1))
            array_meanz = np.tile(np.expand_dims(self.mean_z, axis=0),(N,1))
            array_stdy = np.tile(np.expand_dims(self.std_y, axis=0),(N,1))
            array_meany = np.tile(np.expand_dims(self.mean_y, axis=0),(N,1))
            array_stdx = np.tile(np.expand_dims(self.std_x, axis=0),(N,1))
            array_meanx = np.tile(np.expand_dims(self.mean_x, axis=0),(N,1))

            if(len(forwardsim_x_true)==2):  # 本代码运行这个
                #N starting states, one for each of the simultaneous sims
                curr_states=np.tile(forwardsim_x_true[0], (N,1))  # forwardsim_x_true[0]复制N遍，3000遍
                print("初始curr_states.shape:", curr_states.shape)  # (3000, 8)，互相之间每一个2维数组是相同的
                print("初始curr_states:",curr_states)

            else:
                curr_states=np.copy(forwardsim_x_true)
            print("###########################curr_state############", curr_states.shape)   # (3000, 8)
            #advance all N sims, one timestep at a time
            for timestep in range(horizon):  # 17

                #keep track of states for all N sims
                state_list.append(np.copy(curr_states))
#                print ("###currstate###",len(curr_states[1]),"###curr##")
                #make [N x (state,action)] array to pass into NN
#                print ("###timestep###",curr_states.shape,forwardsim_y.shape,array_meanx.shape,array_meany.shape,"###timestep##")
                states_preprocessed = np.nan_to_num(np.divide((curr_states-array_meanx), array_stdx))
                print("states_preprocessed.shape:", states_preprocessed.shape)  # (3000, 8)
                #print("states_preprocessed:",states_preprocessed)


                print("forwardsim_y.shape为：", forwardsim_y.shape)  # (3000, 17, 2)
                #print("forwardsim_y为：", forwardsim_y)  # 除了初始值之外，每一个2维数组里面的一维元素是相同的

                # 提取3维数组里面的，每一个2维数组的对应第timestep行元素
                print("forwardsim_y[:,timestep,:]为：",forwardsim_y[:,timestep,:])

                actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:,timestep,:]-array_meany), array_stdy))
                print("actions_preprocessed.shape:", actions_preprocessed.shape)  # (3000, 2)
                #print("actions_preprocessed:",actions_preprocessed)


#                print (states_preprocessed.shape,actions_preprocessed.shape)
                inputs_list= np.concatenate((states_preprocessed[:,3:8], actions_preprocessed), axis=1)
#                print (inputs_list.shape)
#                inputs_list = np.concatenate((curr_states, forwardsim_y))   ###, axis=1)    #####changed  by sun
#                inputs_list = np.array([inputs_list])
#                print (inputs_list.shape,)
#                inputs_list1 = np.concatenate((curr_states, forwardsim_y[:,timestep,:]), axis=1)
#                print(inputs_list1.shape, )
#                print ("#######################################################################################")
                #run the N sims all at once
                model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list})
#                model_output = self.sess.run([self.out], feed_dict={self.x_: inputs_list1})
#                print ("###states_differences",len(model_output),len(model_output[0]),len(model_output[0][0]))
                state_differences = np.multiply(model_output[0],array_stdz[:,0:6])+array_meanz[:,0:6]
                
               
                #update the state info
#                print ("###states_differences############",curr_states.shape,state_differences.shape, len(model_output))
                curr_states = curr_states[:,0:6] + state_differences
                curr_states = np.concatenate((curr_states[:,0:6],np.array([np.sin(curr_states[:,2])]).T,np.array([np.cos(curr_states[:,2])]).T),axis=1)

            #return a list of length = horizon+1... each one has N entries, where each entry is (13,)
            state_list.append(np.copy(curr_states))


        else:
            curr_state = np.copy(forwardsim_x_true) #curr state is of dim NN input
            print("curr_state.shape为：",curr_state.shape)

#            for curr_control in forwardsim_y:
            if(1==1):
                curr_control = forwardsim_y
#                print ("######for curr_state###",curr_state,forwardsim_y,curr_control)
                state_list.append(np.copy(curr_state))
#                curr_control = np.expand_dims(curr_control, axis=0)

                #subtract mean and divide by standard deviation
                curr_state_preprocessed = curr_state - self.mean_x
#                print ("###delfmean##",self.mean_x.shape,self.mean_y.shape)
                curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/self.std_x)
                curr_control_preprocessed = curr_control - self.mean_y
                curr_control_preprocessed = np.nan_to_num(curr_control_preprocessed/self.std_y)
                #inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

#                print ("###inputs_preprosessed",curr_state.shape,  curr_state_preprocessed.shape,   curr_control_preprocessed.shape)

                inputs_preprocessed = np.concatenate((curr_state_preprocessed[3:8], curr_control_preprocessed), axis=0)
                inputs_preprocessed = np.array([inputs_preprocessed]) 

#                print ("###inputs_preprosessed",curr_state.shape,curr_state_preprocessed.shape, curr_control_preprocessed.shape,inputs_preprocessed.shape)
                        #run through NN to get prediction
                model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed}) 

                #multiply by std and add mean back in
                state_differences = np.multiply(model_output[0] ,self.std_z[0:6])+self.mean_z[0:6]
#                state_differences= (model_output[0][0]*self.std_z)+self.mean_z

                #update the state info
#                print ("###states_differences",state_differences.shape,model_output.shape,self.curr_nn_output.shape)
                curr_state = curr_state[0:6] + state_differences
#                print (curr_state.shape,state_differences.shape,np.array([np.sin(curr_state[:,2])]).T.shape)
                curr_state = np.concatenate((curr_state[0:6],np.array([np.sin(curr_state[:,2])]).T,np.array([np.cos(curr_state[:,2])]).T),axis=1)
                #copy the state info
                curr_state= np.copy(curr_state)

            state_list.append(np.copy(curr_state))
              
        return state_list
