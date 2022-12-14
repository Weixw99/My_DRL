
#imports
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle
import copy
import os
import sys
from six.moves import cPickle
import yaml
import argparse
import json
import math

#my imports
from trajectories import make_trajectory
from trajectories import get_trajfollow_params
from dynamics_model import Dyn_Model
from mpc_controller import MPCController
from usv import Vehicle

def main():

    #################################################
    ############ commandline arguments ##############
    #################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='usv_trajfollow')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--use_existing_training_data', action="store_true", dest='use_existing_training_data', default=False)
    parser.add_argument('--use_existing_dynamics_model', action="store_true", dest='use_existing_dynamics_model', default=False)

    parser.add_argument('--desired_traj_type', type=str, default='straight') #straight, left_turn, right_turn, u_turn, backward, forward_backward
    parser.add_argument('--num_rollouts_save_for_mf', type=int, default=20)

    parser.add_argument('--print_minimal', action="store_true", dest='print_minimal', default=False)
    args = parser.parse_args()

#print(args)
#    python abc.py
#    Namespace(desired_traj_type='straight', might_render=False, num_rollouts_save_for_mf=60, perform_forwardsim_for_vis=False, print_minimal=False, run_num=0, seed=0, use_existing_dynamics_model=False, use_existing_training_data=False, visualize_MPC_rollout=False, yaml_file='ant_forward')

    ########################################
    ######### params from yaml file ########
    ########################################

    #load in parameters from specified file

    yaml_path = os.path.abspath('yaml_files/'+args.yaml_file+'.yaml')
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)

    #save params from specified file

    follow_trajectories = params['follow_trajectories']
    args.run_num = params['args.run_num']
    args.use_existing_training_data = params['args.use_existing_training_data']
    args.use_existing_dynamics_model = params['args.use_existing_dynamics_model']

    #data collection
    use_threading = params['data_collection']['use_threading']
    num_rollouts_train = params['data_collection']['num_rollouts_train']
    num_rollouts_val = params['data_collection']['num_rollouts_val']
    #dynamics model
    num_fc_layers = params['dyn_model']['num_fc_layers']
    depth_fc_layers = params['dyn_model']['depth_fc_layers']
    batchsize = params['dyn_model']['batchsize']
    lr = params['dyn_model']['lr']
    nEpoch = params['dyn_model']['nEpoch']
    fraction_use_new = params['dyn_model']['fraction_use_new']
    #controller
    horizon = params['controller']['horizon']
    num_control_samples = params['controller']['num_control_samples']
    horizons = params['controller']['horizons']
    ##if(which_agent==1):
    ##    if(args.desired_traj_type=='straight'):
    ##        num_control_samples=3000
    #aggregation
    num_aggregation_iters = params['aggregation']['num_aggregation_iters']
    num_trajectories_for_aggregation = params['aggregation']['num_trajectories_for_aggregation']
    rollouts_forTraining = params['aggregation']['rollouts_forTraining']
    #noise
    make_aggregated_dataset_noisy = params['noise']['make_aggregated_dataset_noisy']
    make_training_dataset_noisy = params['noise']['make_training_dataset_noisy']
    noise_actions_during_MPC_rollouts = params['noise']['noise_actions_during_MPC_rollouts']
    #steps
    dt_steps = params['steps']['dt_steps']
    steps_per_episode = params['steps']['steps_per_episode']
    steps_per_rollout_train = params['steps']['steps_per_rollout_train']
    steps_per_rollout_val = params['steps']['steps_per_rollout_val']
    #saving
    min_rew_for_saving = params['saving']['min_rew_for_saving']
    #generic
    ##visualize_True = params['generic']['visualize_True']
    ##visualize_False = params['generic']['visualize_False']
    #from args
    print_minimal= args.print_minimal


    ########################################
    ### make directories for saving data ###
    ########################################
    save_dir = 'run_'+ str(args.run_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'/losses')
        os.makedirs(save_dir+'/models')
        os.makedirs(save_dir+'/saved_forwardsim')
        os.makedirs(save_dir+'/saved_trajfollow')
        os.makedirs(save_dir+'/training_data')

    ########################################
    ############## set vars ################
    ########################################

    #set seeds
    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    #data collection, either with or without multi-threading
    if(use_threading):
        from collect_samples_threaded import CollectSamples
    else:
        from collect_samples import CollectSamples


    ##x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index = get_indices(which_agent)
    ####################################################################
    ###########################initial vehicle model####################
    ####################################################################

    x_index, y_index, fi_index, u_index, v_index, r_index, Tu_index, Tv_index, Tr_index = [0,1,2,3,4,5,6,7,8]
    tf_datatype = tf.float64
    noiseToSignal = 0.01

    # n is noisy, c is clean... 1st letter is what action's executed and 2nd letter is what action's aggregated
    actions_ag='nc'

    #################################################
    ######## save param values to a file ############
    #################################################
###################################################################################################
#######                  settings                ##################################################
###################################################################################################
#    args.use_existing_training_data = 1
    args.desired_traj_type = "circle"
    follow_trajectories = "usv_trajfollow"
#    args.use_existing_dynamics_model = 0


    param_dict={}
    ##param_dict['which_agent']= which_agent
    param_dict['use_existing_training_data']= str(args.use_existing_training_data)
    param_dict['desired_traj_type']= args.desired_traj_type
    ##param_dict['visualize_MPC_rollout']= str(args.visualize_MPC_rollout)
    param_dict['num_rollouts_save_for_mf']= args.num_rollouts_save_for_mf
    param_dict['seed']= args.seed
    param_dict['follow_trajectories']= str(follow_trajectories)
    param_dict['use_threading']= str(use_threading)
    param_dict['num_rollouts_train']= num_rollouts_train
    param_dict['num_fc_layers']= num_fc_layers
    param_dict['depth_fc_layers']= depth_fc_layers
    param_dict['batchsize']= batchsize
    param_dict['lr']= lr
    param_dict['nEpoch']= nEpoch
    param_dict['fraction_use_new']= fraction_use_new
    param_dict['horizon']= horizon
    param_dict['horizons'] = horizons
    param_dict['num_control_samples']= num_control_samples
    param_dict['num_aggregation_iters']= num_aggregation_iters
    param_dict['num_trajectories_for_aggregation']= num_trajectories_for_aggregation
    param_dict['rollouts_forTraining']= rollouts_forTraining
    param_dict['make_aggregated_dataset_noisy']= str(make_aggregated_dataset_noisy)
    param_dict['make_training_dataset_noisy']= str(make_training_dataset_noisy)
    param_dict['noise_actions_during_MPC_rollouts']= str(noise_actions_during_MPC_rollouts)
    param_dict['dt_steps']= dt_steps
    param_dict['steps_per_episode']= steps_per_episode
    param_dict['steps_per_rollout_train']= steps_per_rollout_train
    param_dict['steps_per_rollout_val']= steps_per_rollout_val
    param_dict['min_rew_for_saving']= min_rew_for_saving
    param_dict['x_index']= x_index
    param_dict['y_index']= y_index
    param_dict['tf_datatype']= str(tf_datatype)
    param_dict['noiseToSignal']= noiseToSignal

    with open(save_dir+'/params.pkl', 'wb') as f:
        pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(save_dir+'/params.txt', 'w') as f:
        f.write(json.dumps(param_dict))

    #################################################
    ### initialize the experiment
    #################################################

    if(not(print_minimal)):
        print("\n#####################################")
        print("Initializing environment")
        print("#####################################\n")

    #create env
    #env, dt_from_xml= create_env(which_agent)

    #create random policy for data collection
    #random_policy = Policy_Random(env)

    #################################################
    ### set GPU options for TF
    #################################################

    gpu_device = 0
    gpu_frac = 0.3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
#####dong
    with tf.Session(config=config) as sess:
        #################################################
        ### deal with data
        #################################################

        if(args.use_existing_training_data):
            if(not(print_minimal)):
                print("\n#####################################")
                print("Retrieving training data & policy from saved files")
                print("#####################################\n")

            dataX= np.load(save_dir + '/training_data/dataX.npy') # input1: state
            dataY= np.load(save_dir + '/training_data/dataY.npy') # input2: control
            dataZ= np.load(save_dir + '/training_data/dataZ.npy') # output: nextstate-state
            states_val= np.load(save_dir + '/training_data/states_val.npy')
            controls_val= np.load(save_dir + '/training_data/controls_val.npy')
#            forwardsim_x_true= np.load(save_dir + '/training_data/forwardsim_x_true.npy')
#            forwardsim_y= np.load(save_dir + '/training_data/forwardsim_y.npy')

        else:

            if(not(print_minimal)):
                print("\n#####################################")
                print("Performing rollouts to collect training data")
                print("#####################################\n")
            actions = []  ###input1 NN
            front_states = []  ###input2  NN
            next_states = []  ###output
            num_rollouts_train =700
            dt_steps = 100
            pi = math.pi
            x1 = 0
            x2 = 0
            x3 = 0 * pi
            x4 = 0.2
            x5 = 0
            x6 = 0.1
            x31 = math.sin(x3)
            x32 = math.cos(x3)
            for i in range(num_rollouts_train * dt_steps) :
                  x7 = np.random.uniform(-2,2,1)[0]     #Tu
                  x8 = 0
                  x9 = np.random.uniform(-1.5,1.5,1)[0] #Tr
                  T = np.array([x7,x9])
                  front_pos = np.array([x1,x2,x3,x4,x5,x6,x31,x32])
                  actions.append(T)  ####liebiao pinjie
                  front_states.append(front_pos)
                  eta_dot , nu_dot  = Vehicle(x1,x2,x3,x4,x5,x6,x7,x8,x9,i)

                  eta_dot = eta_dot.tolist()
                  nu_dot = nu_dot.tolist()
#                      x3_dot =[np.array([sin_dot]),np.array([cos_dot])]
                  out = np.concatenate(np.concatenate((eta_dot,nu_dot),axis=0),axis=0)##1hang,6lie

                  [x1,x2,x3,x4,x5,x6] = out
                  x31 = math.sin(x3)
                  x32 = math.cos(x3)
                  out = np.concatenate((out,[x31,x32]),axis=0) ####x,y,fi,u,v,r,sinfi,cosfi
                  out = np.array(out)

                  next_states.append(out)
            print ("ALLDONE!")
            if(not(print_minimal)):
                print("\n#####################################")
                print("Performing rollouts to collect validation data")
                print("#####################################\n")

            start_validation_rollouts = time.time()
#            states_val, controls_val, _, _ = perform_rollouts(random_policy, num_rollouts_val, steps_per_rollout_val, visualize_False,
#                                                            CollectSamples, env, which_agent, dt_steps, dt_from_xml, follow_trajectories)
            controls_val= []  ###input1 NN
            states_val = []  ###input2  NN
#            next_states_val = []  ###output
#            num_rollouts_train =500
#            dt_steps = 100
            for rollout_number in range(num_rollouts_val) :
                  pi = math.pi
                  x1_val = 0
                  x2_val = 0
                  x3_val = 0 * pi
                  x4_val = 0.2
                  x5_val = 0
                  x6_val= 0.1
                  x31_val = math.sin(x3_val)
                  x32_val = math.cos(x3_val)
                  actions_val_step = []
                  states_val_step = []
#                  next_states_val_step = []
                  for step_num in range(steps_per_rollout_val) :
                        x7_val = np.random.uniform(-2,2,1)[0]     #Tu
                        x8_val = 0
                        x9_val = np.random.uniform(-1.5,1.5,1)[0] #Tr
                        T = np.array([x7_val,x9_val])
                        states_pos = np.array([x1_val,x2_val,x3_val,x4_val,x5_val,x6_val,x31_val,x32_val])
                        actions_val_step.append(T)  ####liebiao pinjie
                        states_val_step.append(states_pos)
                        eta_dot , nu_dot  = Vehicle(x1_val,x2_val,x3_val,x4_val,x5_val,x6_val,x7_val,x8_val,x9_val,step_num)

                        eta_dot = eta_dot.tolist()
                        nu_dot = nu_dot.tolist()
#                      x3_dot =[np.array([sin_dot]),np.array([cos_dot])]
                        out = np.concatenate(np.concatenate((eta_dot,nu_dot),axis=0),axis=0)##1hang,6lie

                        [x1_val,x2_val,x3_val,x4_val,x5_val,x6_val] = out
                        x31_val = math.sin(x3_val)
                        x32_val = math.cos(x3_val)
                        out = np.concatenate((out,[x31_val,x32_val]),axis=0) ####x,y,fi,u,v,r,sinfi,cosfi
                        out = np.array(out)
                        states_pos = np.copy(out)
#                        next_states.append(out)
                  states_val_step = np.array(states_val_step)
                  actions_val_step = np.array(actions_val_step)
                  states_val.append(states_val_step)
                  controls_val.append(actions_val_step)
            print ("ALLDONE!")
            if(not(print_minimal)):
                print("\n#####################################")
                print("Convert from env observations to NN 'states' ")
                print("#####################################\n")

            #training
#            states = from_observation_to_usablestate(states, which_agent, False)
            #validation
#            states_val = from_observation_to_usablestate(states_val, which_agent, False)
            states_val = np.array(states_val)

            print ("ALLDONE!")
            if(not(print_minimal)):
                print("\n#####################################")
                print("Data formatting: create inputs and labels for NN ")
                print("#####################################\n")

            dataX = np.array(front_states)
            dataY = np.array( actions )
#            differences = []
#            for states_in_single_rollout in front_states:
#                output = states_in_single_rollout[1:states_in_single_rollout.shape[0], :] \
#                         - states_in_single_rollout[0:states_in_single_rollout.shape[0] - 1, :]
#                differences.append(output)
#            output = np.concatenate(differences, axis=0)
            dataZ = np.array( (np.array(next_states) - np.array(front_states)) )
#            print (dataX[0],dataX[1],dataX[2])
#            print("\n#####################################")
#            print (dataY[0],dataY[1])
#            print("\n#####################################")
#            print (dataZ[0],dataZ[1])
            if(not(print_minimal)):
                print("\n#####################################")
                print("Add noise")
                print("#####################################\n")

            #add a little dynamics noise (next state is not perfectly accurate, given correct state and action)
#            if(make_training_dataset_noisy):
#                dataX = add_noise(dataX, noiseToSignal)
#                dataZ = add_noise(dataZ, noiseToSignal)

###            if(not(print_minimal)):
###                print("\n#####################################")
###                print("Perform rollout & save for forward sim")
###                print("#####################################\n")
###
###            states_forwardsim_orig, controls_forwardsim, _,_ = perform_rollouts(random_policy, 1, 100,
###                                                                            visualize_False, CollectSamples,
###                                                                            env, which_agent, dt_steps,
###                                                                            dt_from_xml, follow_trajectories)
###            states_forwardsim = np.copy(from_observation_to_usablestate(states_forwardsim_orig, which_agent, False))
###            forwardsim_x_true, forwardsim_y = generate_training_data_inputs(states_forwardsim, controls_forwardsim)

            if(not(print_minimal)):
                print("\n#####################################")
                print("Saving data")
                print("#####################################\n")

            np.save(save_dir + '/training_data/dataX.npy', dataX)
            np.save(save_dir + '/training_data/dataY.npy', dataY)
            np.save(save_dir + '/training_data/dataZ.npy', dataZ)
            np.save(save_dir + '/training_data/states_val.npy', states_val)
            np.save(save_dir + '/training_data/controls_val.npy', controls_val)
#            np.save(save_dir + '/training_data/forwardsim_x_true.npy', forwardsim_x_true)
#            np.save(save_dir + '/training_data/forwardsim_y.npy', forwardsim_y)
        print ("ALLDONE!!!!")
        if(not(print_minimal)):
            print("Done getting data.")
            print("dataX dim: ", dataX.shape)

        #################################################
        ### init vars
        #################################################

        counter_agg_iters=0  ###the times of already run
        training_loss_list=[]
        forwardsim_score_list=[]
        old_loss_list=[]
        new_loss_list=[]
        errors_1_per_agg=[]
        errors_5_per_agg=[]
        errors_10_per_agg=[]
        errors_50_per_agg=[]
        errors_100_per_agg=[]
        list_avg_rew=[]
        list_num_datapoints=[]
        dataX_new = np.zeros((0,dataX.shape[1]))
        dataY_new = np.zeros((0,dataY.shape[1]))
        dataZ_new = np.zeros((0,dataZ.shape[1]))

        #################################################
        ### preprocess the old training dataset
        #################################################

        if(not(print_minimal)):
            print("\n#####################################")
            print("Preprocessing 'old' training data")
            print("#####################################\n")

        #every component (i.e. x position) should become mean 0, std 1   ######all components
        ####every row become mean=0,std=1
        mean_x = np.mean(dataX, axis = 0)
        dataX = dataX - mean_x
        std_x = np.std(dataX, axis = 0)
        dataX = np.nan_to_num(dataX/std_x)

        mean_y = np.mean(dataY, axis = 0)
        dataY = dataY - mean_y
        std_y = np.std(dataY, axis = 0)
        dataY = np.nan_to_num(dataY/std_y)

        mean_z = np.mean(dataZ, axis = 0)
        dataZ = dataZ - mean_z
        std_z = np.std(dataZ, axis = 0)
        dataZ = np.nan_to_num(dataZ/std_z)

        ## concatenate state and action, to be used for training dynamics
        inputs = np.concatenate((dataX[:,3:8], dataY), axis=1)
        outputs = np.copy(dataZ[:,0:6])
#        print (inputs[0],outputs[0],inputs[1],outputs[1],inputs[2],outputs[2])
        #doing a render here somehow allows it to not produce an error later
#        might_render= False
#        if(args.visualize_MPC_rollout or args.might_render):
#            might_render=True
#        if(might_render):
#            new_env, _ = create_env(which_agent)
#            new_env.render()

        ##############################################
        ########## THE AGGREGATION LOOP ##############
        ##############################################

        #dimensions
        assert inputs.shape[0] == outputs.shape[0]
        inputSize = inputs.shape[1]
        outputSize = outputs.shape[1]

        #initialize dynamics model
        dyn_model = Dyn_Model(inputSize, outputSize, sess, lr, batchsize, x_index, y_index, num_fc_layers,
                            depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal)



        #create mpc controller
        mpc_controller = MPCController(dyn_model, horizon, steps_per_episode, dt_steps, num_control_samples,
                                        mean_x, mean_y, mean_z, std_x, std_y, std_z, actions_ag, print_minimal, x_index, y_index,
                                        fi_index, u_index, v_index, r_index,horizons)


        #randomly initialize all vars
        sess.run(tf.global_variables_initializer())

#        num_aggregation_iters = 1
        starting_big_loop1 = time.time()

        while(counter_agg_iters<num_aggregation_iters):

            #make saver
            if(counter_agg_iters==0):
                saver = tf.train.Saver(max_to_keep=0)        ############save the trainned model's parameters

            print("\n#####################################")
            print("AGGREGATION ITERATION ", counter_agg_iters)
            print("#####################################\n")

            #save the aggregated dataset used to train during this agg iteration
            np.save(save_dir + '/training_data/dataX_new_iter'+ str(counter_agg_iters) + '.npy', dataX_new)
            np.save(save_dir + '/training_data/dataY_new_iter'+ str(counter_agg_iters) + '.npy', dataY_new)
            np.save(save_dir + '/training_data/dataZ_new_iter'+ str(counter_agg_iters) + '.npy', dataZ_new)

            starting_big_loop = time.time()

            if(not(print_minimal)):
                print("\n#####################################")
                print("Preprocessing 'new' training data")
                print("#####################################\n")

            dataX_new_preprocessed = np.nan_to_num((dataX_new - mean_x)/std_x)
            dataY_new_preprocessed = np.nan_to_num((dataY_new - mean_y)/std_y)
            dataZ_new_preprocessed = np.nan_to_num((dataZ_new - mean_z)/std_z)

            ## concatenate state and action, to be used for training dynamics
            inputs_new = np.concatenate((dataX_new_preprocessed[:,3:8], dataY_new_preprocessed), axis=1)  #####changed by sun
            outputs_new = np.copy(dataZ_new_preprocessed[:,0:6])  ####changed by sun

            if(not(print_minimal)):
                print("\n#####################################")
                print("Training the dynamics model")
                print("#####################################\n")

            #train model or restore model
            if(args.use_existing_dynamics_model):
                restore_path = save_dir+'/models/model_aggIter' +str(counter_agg_iters)+ '.ckpt'
#                restore_path = save_dir + '/models/model_aggIter4.ckpt'
#                restore_path = save_dir+ '/models/finalModel.ckpt'
                saver.restore(sess, restore_path)                    #####in run_num
                print("Model restored from ", restore_path)
                training_loss=0
                old_loss=0
                new_loss=0
            else:

                training_loss, old_loss, new_loss = dyn_model.train(inputs, outputs, inputs_new, outputs_new,
                                                                    nEpoch, save_dir, fraction_use_new)

            #how good is model on training data
            training_loss_list.append(training_loss)
            #how good is model on old dataset
            old_loss_list.append(old_loss)
            #how good is model on new dataset
            new_loss_list.append(new_loss)

            print("\nTraining loss: ", training_loss)

            #####################################
            ## Saving model
            #####################################

            save_path = saver.save(sess, save_dir+ '/models/model_aggIter' +str(counter_agg_iters)+ '.ckpt')
#            save_path = saver.save(sess, save_dir+ '/models/finalModel.ckpt')
            if(not(print_minimal)):
                print("Model saved at ", save_path)

            #####################################
            ## calculate multi-step validation metrics
            #####################################

            if(not(print_minimal)):
                print("\n#####################################")
                print("Calculating Validation Metrics")
                print("#####################################\n")

            #####################################
            ## init vars for multi-step validation metrics
            #####################################

            validation_inputs_states = []
            labels_1step = []
            labels_5step = []
            labels_10step = []
            labels_50step = []
            labels_100step = []
            controls_100step=[]

            #####################################
            ## make the arrays to pass into forward sim
            #####################################

            for i in range(num_rollouts_val):        ########states_val=[20,333,8]

                length_curr_rollout = states_val[i].shape[0]

                if(length_curr_rollout>100):

                    #########################
                    #### STATE INPUTS TO NN
                    #########################

                    ## take all except the last 100 pts from each rollout
                    validation_inputs_states.append(states_val[i][0:length_curr_rollout-100])

                    #########################
                    #### CONTROL INPUTS TO NN
                    #########################

                    #100 step controls
                    list_100 = []
                    for j in range(100):
                        list_100.append(controls_val[i][0+j:length_curr_rollout-100+j])
                        ##for states 0:x, first apply acs 0:x, then apply acs 1:x+1, then apply acs 2:x+2, etc...
                    list_100=np.array(list_100) #100xstepsx2
                    list_100= np.swapaxes(list_100,0,1) #stepsx100x2
                    controls_100step.append(list_100)

                    #########################
                    #### STATE LABELS- compare these to the outputs of NN (forward sim)
                    #########################
                    labels_1step.append(states_val[i][0+1:length_curr_rollout-100+1])
                    labels_5step.append(states_val[i][0+5:length_curr_rollout-100+5])
                    labels_10step.append(states_val[i][0+10:length_curr_rollout-100+10])
                    labels_50step.append(states_val[i][0+50:length_curr_rollout-100+50])
                    labels_100step.append(states_val[i][0+100:length_curr_rollout-100+100])

            validation_inputs_states = np.concatenate(validation_inputs_states)
            controls_100step = np.concatenate(controls_100step)
            labels_1step = np.concatenate(labels_1step)
            labels_5step = np.concatenate(labels_5step)
            labels_10step = np.concatenate(labels_10step)
            labels_50step = np.concatenate(labels_50step)
            labels_100step = np.concatenate(labels_100step)

            #####################################
            ## pass into forward sim, to make predictions
            #####################################

            many_in_parallel = True
            predicted_100step = dyn_model.do_forward_sim(validation_inputs_states, controls_100step,
                                                        many_in_parallel)
            np.save(save_dir + '/saved_forwardsim/pred_iter' + str(counter_agg_iters) + '.npy', predicted_100step)
            #####################################
            ## Calculate validation metrics (mse loss between predicted and true)
            #####################################

            array_meanx = np.tile(np.expand_dims(mean_x, axis=0),(labels_1step.shape[0],1))
            array_stdx = np.tile(np.expand_dims(std_x, axis=0),(labels_1step.shape[0],1))

            error_1step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[1]-array_meanx,array_stdx))
                                -np.nan_to_num(np.divide(labels_1step-array_meanx,array_stdx))))
            error_5step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[5]-array_meanx,array_stdx))
                                -np.nan_to_num(np.divide(labels_5step-array_meanx,array_stdx))))
            error_10step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[10]-array_meanx,array_stdx))
                                    -np.nan_to_num(np.divide(labels_10step-array_meanx,array_stdx))))
            error_50step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[50]-array_meanx,array_stdx))
                                    -np.nan_to_num(np.divide(labels_50step-array_meanx,array_stdx))))
            error_100step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[100]-array_meanx,array_stdx))
                                    -np.nan_to_num(np.divide(labels_100step-array_meanx,array_stdx))))
            print("Multistep error values: ", error_1step, error_5step, error_10step, error_50step, error_100step,"\n")

            errors_1_per_agg.append(error_1step)
            errors_5_per_agg.append(error_5step)
            errors_10_per_agg.append(error_10step)
            errors_50_per_agg.append(error_50step)
            errors_100_per_agg.append(error_100step)

            #####################################
            ## Perform 1 forward simulation, for visualization purposes (compare predicted traj vs true traj)
            #####################################

#            if(args.perform_forwardsim_for_vis):
#                if(not(print_minimal)):
#                    print("\n#####################################")
#                    print("Performing a forward sim of the learned model. using pre-saved dataset. just for visualization")
#                    print("#####################################\n")
#            
#                #for a given set of controls,
#                #compare sim traj vs. learned model's traj 
#                #(dont expect this to be good cuz error accum)
#                many_in_parallel = False
#                forwardsim_x_pred = dyn_model.do_forward_sim(forwardsim_x_true, forwardsim_y, many_in_parallel, env, which_agent)    
#                forwardsim_x_pred = np.array(forwardsim_x_pred)
#
#                # save results of forward sim
#                np.save(save_dir + '/saved_forwardsim/forwardsim_states_true_'+str(counter_agg_iters)+'.npy', forwardsim_x_true)
#                np.save(save_dir + '/saved_forwardsim/forwardsim_states_pred_'+str(counter_agg_iters)+'.npy', forwardsim_x_pred)

            #####################################
            ######## EXECUTE CONTROLLER #########
            #####################################

            if(not(print_minimal)):
                print("##############################################")
                print("#### Execute the controller to follow desired trajectories")
                print("##############################################\n")

            ###################################################################
            ### Try to follow trajectory... collect rollouts
            ###################################################################

            #init vars
            list_rewards=[]
            starting_states=[]
            selected_multiple_u = []
            resulting_multiple_x = []

            #get parameters for trajectory following
            horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, r_penalty_factor, u_penalty_factor = get_trajfollow_params(args.desired_traj_type)

#            if(follow_trajectories==False):
#                desired_snake_headingInit=0

            for rollout_num in range(num_trajectories_for_aggregation):

                if(not(print_minimal)):
                    print("\nPerforming MPC rollout #", rollout_num)

#                #reset env and set the desired traj 
#                if(which_agent==2):
#                    starting_observation, starting_state = env.reset(evaluating=True, returnStartState=True, isSwimmer=True)
#                else:
#                    starting_observation, starting_state = env.reset(evaluating=True, returnStartState=True)
#                #start swimmer heading in correct direction
#                if(which_agent==2):
#                starting_state[2] = desired_snake_headingInit
#                    starting_observation, starting_state = env.reset(starting_state, returnStartState=True)

                x_taj, y_taj, fi_taj, u_taj, v_taj, r_taj = [0,0,0,0,0,0]
                starting_state_taj = np.array([ x_taj, y_taj, fi_taj, u_taj, v_taj, r_taj])      #####x,y,fi,u,v,r

#                x_mpc, y_mpc, fi_mpc, u_mpc, v_mpc, r_mpc = [1.0, 15, -math.pi, 0, 0, 0]
                x_mpc, y_mpc, fi_mpc, u_mpc, v_mpc, r_mpc = [15, -1, 0.0, 0, 0, 0]
#                x_mpc, y_mpc, fi_mpc, u_mpc, v_mpc, r_mpc = [-20.02 ,0 ,-math.pi/2 ,0 ,0 ,0 ]
#                x_mpc, y_mpc, fi_mpc, u_mpc, v_mpc, r_mpc = [-0.02, -20, 0, 0, 0, 0]
#                starting_observation = np.array([ x_mpc, y_mpc, fi_mpc, u_mpc, v_mpc, r_mpc]) ####x,y,fi,u,v,r
               
                #desired trajectory to follow`
#                inputs = np.concatenate((dataX[:,3:8], dataY), axis=1)
#                outputs = np.copy(dataZ[:,0:6])

                starting_observation_NNinput = np.array([ x_mpc, y_mpc, fi_mpc, u_mpc, v_mpc, r_mpc, math.sin(fi_mpc), math.cos(fi_mpc)])  ####x,y,fi,u,v,r,sinfi,cosfi
#                starting_observation_NNinput = from_observation_to_usablestate(starting_observation, which_agent, True)
#                print ("##trajory##",args.desired_traj_type)
                desired_x = make_trajectory(args.desired_traj_type, starting_state_taj)
                print ("###desired##",desired_x.shape,desired_x[0],desired_x[1],desired_x[30])
                #perform 1 MPC rollout
                #depending on follow_trajectories, either move forward or follow desired_traj_type
                if(noise_actions_during_MPC_rollouts):
                    curr_noise_amount = 0.005
                else:
                    curr_noise_amount=0
                resulting_x, selected_u  = mpc_controller.perform_rollout(starting_observation_NNinput, desired_x, 
                                                                        follow_trajectories, horiz_penalty_factor, 
                                                                        forward_encouragement_factor, heading_penalty_factor, r_penalty_factor, u_penalty_factor,
                                                                        noise_actions_during_MPC_rollouts, curr_noise_amount)

                #save info from MPC rollout
#                list_rewards.append(ep_rew)
                selected_multiple_u.append(selected_u)
                resulting_multiple_x.append(resulting_x)
#                starting_states.append(starting_state)

#            if(args.visualize_MPC_rollout):
#                input("\n\nPAUSE BEFORE VISUALIZATION... Press Enter to continue...")
#                for vis_index in range(num_trajectories_for_aggregation):
#                    visualize_rendering(starting_states[vis_index], selected_multiple_u[vis_index], env, dt_steps, dt_from_xml, which_agent)

            #bookkeeping
#            avg_rew = np.mean(np.array(list_rewards))
#            std_rew = np.std(np.array(list_rewards))
#            print("############# Avg reward for ", num_trajectories_for_aggregation, " MPC rollouts: ", avg_rew)
#            print("############# Std reward for ", num_trajectories_for_aggregation, " MPC rollouts: ", std_rew)
#            print("############# Rewards for the ", num_trajectories_for_aggregation, " MPC rollouts: ", list_rewards)
            print ("ALLDONE MPC!!!!!!!!!")
            #save pts_used_so_far + performance achieved by those points
            list_num_datapoints.append(dataX.shape[0]+dataX_new.shape[0])
#            list_avg_rew.append(avg_rew)

            ##############################
            ### Aggregate data
            ##############################

            if (counter_agg_iters < (num_aggregation_iters - 1)):
                ##############################
                ### aggregate some rollouts into training set
                ##############################

                x_array = np.array(resulting_multiple_x)[:, 0:(rollouts_forTraining + 1),
                          :]  # ?????????rollouts_forTraining + 1???,????????????[1,rollouts_forTraining + 1,8]
                x_array = np.squeeze(x_array)  ####??????????????????[rollouts_forTraining + 1,8]
                #                if (which_agent == 6 or which_agent == 1):
                #                    u_array = np.array(selected_multiple_u)[0:(rollouts_forTraining + 1)]
                #                else:
                print(np.array(selected_multiple_u).shape)
#                print(np.array(selected_multiple_u))
#                print (x_array)
                u_array = np.squeeze(np.array(selected_multiple_u))[0:(rollouts_forTraining),
                          :]  ###???????????????[rollouts_forTraining ,2]
                print("###uarray,xarray###", x_array.shape, u_array.shape, np.array(resulting_multiple_x).shape,
                      np.array(selected_multiple_u).shape)
                dataX_new1 = np.zeros((0, dataX.shape[1]))  # []
                dataY_new1 = np.zeros((0, dataY.shape[1]))
                dataZ_new1 = np.zeros((0, dataZ.shape[1]))
                if (steps_per_episode <= rollouts_forTraining):
                    rollouts_forTraining = steps_per_episode
                for i in range(rollouts_forTraining):
                    x0 = x_array[i]  # [N+1, NN_inp]
                    x1 = x_array[i + 1]
                    u = u_array[i]  # [N, actionSize]

                    newDataX = np.copy([x0])
                    newDataY = np.copy([u])
                    newDataZ = np.copy([x1 - x0])
                    # make this new data a bit noisy before adding it into the dataset
                    #                   if (make_aggregated_dataset_noisy):
                    #                       newDataX = add_noise(newDataX, noiseToSignal)
                    #                       newDataZ = add_noise(newDataZ, noiseToSignal)

                    # the actual aggregation
                    dataX_new1 = np.concatenate((dataX_new1, newDataX))
                    dataY_new1 = np.concatenate((dataY_new1, newDataY))
                    dataZ_new1 = np.concatenate((dataZ_new1, newDataZ))

                dataX_new = np.copy(dataX_new1)
                dataY_new = np.copy(dataY_new1)
                dataZ_new = np.copy(dataZ_new1)
                print(dataX_new.shape, dataY_new.shape, dataZ_new.shape)
  

            #save trajectory following stuff (aka trajectory taken) for plotting
#            np.save(save_dir + '/saved_trajfollow/startingstate_iter' + str(counter_agg_iters) +'.npy', starting_state)
            np.save(save_dir + '/saved_trajfollow/control_iter' + str(counter_agg_iters) +'.npy', selected_u)
            np.save(save_dir + '/saved_trajfollow/true_iter' + str(counter_agg_iters) +'.npy', desired_x)
            np.save(save_dir + '/saved_trajfollow/pred_iter' + str(counter_agg_iters) +'.npy', np.array(resulting_multiple_x))

            #bookkeeping
            if(not(print_minimal)):
                print("\n\nDONE WITH BIG LOOP ITERATION ", counter_agg_iters ,"\n\n")
                print("training dataset size: ", dataX.shape[0] + dataX_new.shape[0])
#                if(len(full_states_list)>0):
#                    print("validation dataset size: ", np.concatenate(full_states_list).shape[0])
                print("Time taken: {:0.2f} s\n\n".format(time.time()-starting_big_loop))
            counter_agg_iters= counter_agg_iters+1

            #save things after every agg iteration
            np.save(save_dir + '/errors_1_per_agg.npy', errors_1_per_agg)
            np.save(save_dir + '/errors_5_per_agg.npy', errors_5_per_agg)
            np.save(save_dir + '/errors_10_per_agg.npy', errors_10_per_agg)
            np.save(save_dir + '/errors_50_per_agg.npy', errors_50_per_agg)
            np.save(save_dir + '/errors_100_per_agg.npy', errors_100_per_agg)
            np.save(save_dir + '/avg_rollout_rewards_per_agg.npy', list_avg_rew)
            np.save(save_dir + '/losses/list_training_loss.npy', training_loss_list) 
            np.save(save_dir + '/losses/list_old_loss.npy', old_loss_list)
            np.save(save_dir + '/losses/list_new_loss.npy', new_loss_list)
        print("Time taken: {:0.2f} s\n\n".format(time.time() - starting_big_loop1))
        print("ALL DONE.")

        return

if __name__ == '__main__':
    main()
