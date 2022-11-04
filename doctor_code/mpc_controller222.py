import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import copy
from six.moves import cPickle
from rllab.misc import tensor_utils
from data_manipulation import from_observation_to_usablestate
from reward_functions import RewardFunctions
from usv import Vehicle
class MPCController:

#    def __init__(self, dyn_model, horizon, steps_per_episode, dt_steps, num_control_samples,
#                 actions_ag, print_minimal, x_index, y_index, fi_index,
#                u_index, v_index, r_index, Tu_index, Tv_index, Tr_index):

    def __init__(self, dyn_model, horizon, steps_per_episode, dt_steps, num_control_samples, 
                mean_x, mean_y, mean_z, std_x, std_y, std_z, actions_ag, print_minimal, x_index, y_index, fi_index, u_index, 
                v_index, r_index,horizons):

        #init vars
#        self.env=copy.deepcopy(env_inp)
        self.N = num_control_samples
#        self.which_agent = which_agent
        self.horizon = horizon
        self.horizons = horizons
        self.dyn_model = dyn_model
        self.steps_per_episode = steps_per_episode 
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.x_index = x_index
        self.y_index = y_index
#        self.z_index = z_index
        self.fi_index = fi_index
        self.u_index = u_index   
        self.v_index = v_index
        self.r_index = r_index
#        self.Tu_index = Tu_index  
#        self.Tv_index = Tv_index
#        self.Tr_index = Tr_index             #   velocity
#        self.orientation_index = orientation_index   # orientation方向
        self.actions_ag = 'nn'
        self.print_minimal = print_minimal
        self.reward_functions = RewardFunctions( self.x_index, self.y_index,  self.fi_index,
                                                self.u_index, self.v_index, self.r_index)

    def perform_rollout(self, starting_observation_NNinput, desired_states, follow_trajectories, 
                        horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, r_penalty_factor, u_penalty_factor, noise_actions, noise_amount):

       # 在main里面第799行，starting_observation_NNinput为一维数组（1，8），desired_states为（6500,3）


        #lists for saving info
        traj_taken=[] #list of states that go into NN  
        actions_taken=[]
        observations = [] #list of observations (direct output of the env)
        rewards = []
#        agent_infos = []
#        env_infos = []


        #init vars
        stop_taking_steps = False
        total_reward_for_episode = 0
        step=0
        curr_line_segment = 0

        self.horiz_penalty_factor = horiz_penalty_factor
        self.forward_encouragement_factor = forward_encouragement_factor
        self.heading_penalty_factor = heading_penalty_factor
        self.r_penalty_factor = r_penalty_factor
        self.u_penalty_factor = u_penalty_factor
        #extend the list of desired states so you don't run out
        temp = np.tile(np.expand_dims(desired_states[-1], axis=0), (10,1)) ######(10,2) (x,y) THE LAST STATE
#        print ("###temp##",temp.shape)
        # (6510,3)
        self.desired_states = np.concatenate((desired_states, temp)) ####swimmer straight(68,2),leftturn(18,2)
#        print ("self########",self.desired_states.shape)
        #reset env to the given full env state
#        if(self.which_agent==5):
#            self.env.reset()
#        else:
#            self.env.reset(starting_fullenvstate)

        #current observation
#        observations = np.copy(starting_observation)                ####  mpc states: x,y,fi,u,v,r
        #current observation in the right format for NN
       # curr_state为(1,8)
        curr_state = np.copy(starting_observation_NNinput)  #### x,y,fi,u,v,r,sinfi,cosfi  (8,)
        traj_taken.append(curr_state)
#        observations.append(curr_state)
#        observations = np.copy(curr_state)
        #select task or reward func
        reward_func = self.reward_functions.get_reward_func(follow_trajectories, self.desired_states, horiz_penalty_factor, 
                                                            forward_encouragement_factor, heading_penalty_factor, r_penalty_factor, u_penalty_factor)


        accumulate_reward = 0
        list_accumulate_reward = []
        #take steps according to the chosen task/reward function
        while(stop_taking_steps==False):

            #get optimal action
            # # 初始curr_line_segment=0,curr_state=(8,)
            best_score,best_action, best_sim_number, best_sequence, moved_to_next, best_curr_seg = self.get_action(curr_state, curr_line_segment, reward_func)
            print ("########best_action#######",best_action)
            #advance which line segment we are on

            accumulate_reward += best_score
            print("看看accumulate_reward的变化： ", accumulate_reward)

            #best_scores = best_scores/100
            list_accumulate_reward.append(best_score)






            if(follow_trajectories):
 ##3-13               if(moved_to_next[best_sim_number]==1):
                if(1==1):
                    curr_line_segment +=self.horizons     # ?????
#                    curr_line_segment = best_curr_seg
                    print ("STEP ",step)
                    print("MOVED ON TO LINE SEGMENT ", curr_line_segment, desired_states[curr_line_segment])

            #noise the action
            action_to_take= np.copy(best_action)
             
            #whether to execute noisy or clean actions
#            if(self.actions_ag=='nn'):
#                noise_actions=True
#            if(self.actions_ag=='nc'):
#                noise_actions=True
#            if(self.actions_ag=='cc'):
#                noise_actions=False

#            clean_action = np.copy(action_to_take)
#            if(noise_actions):
#                noise = noise_amount * npr.normal(size=action_to_take.shape)#
#                action_to_take = action_to_take + noise
#                action_to_take=np.clip(action_to_take, -1,1)
#                action_to_take[0] = np.clip(action_to_take[0], -1, 1)
#                action_to_take[1] = np.clip(action_to_take[1], -1, 1)
            next_states = []
            print(curr_state[0])
            (x1,x2,x3,x4,x5,x6) = curr_state[0:6]

            for step_num in range(self.horizons):  # 10
                (x7, x9) = action_to_take
                x8 = 0
                eta_dot, nu_dot = Vehicle(x1,x2,x3,x4,x5,x6,x7,x8,x9,step+step_num)
                eta_dot1 = eta_dot.tolist()
                nu_dot1 = nu_dot.tolist()
                out = np.concatenate(np.concatenate((eta_dot1, nu_dot1), axis=0), axis=0)
                [x1, x2, x3, x4, x5, x6] = out
                x31 = math.sin(x3)
                x32 = math.cos(x3)
                out = np.concatenate((out, [x31, x32]), axis=0)  ####x,y,fi,u,v,r,sinfi,cosfi
#                out = np.array(out)

                next_states.append(out)
                actions_taken.append(np.array([action_to_take]))
 #               print("###nextstate###", next_states[0:6])
            next_state = next_states[self.horizons-1]
            print ("###nextstate###",next_state[0:6])

#            #execute the action
#            next_state, rew, done, env_info = self.env.step(action_to_take, collectingInitialData=False)
#############################            many_in_parallel = False
#            print ("####curr_####",curr_state.shape)
############################            next_state = self.dyn_model.do_forward_sim(curr_state, action_to_take, many_in_parallel)   ###list
            #check if done
#            if(done):
#                stop_taking_steps=True
#            else:
#                #save things
###            curr_state = next_state
###            observations.append(next_state)
#                rewards.append(rew)
#                env_infos.append(env_info)
#                total_reward_for_episode += rew

#                #whether to save clean or noisy actions
#            if(self.actions_ag=='nn'):
#                for i in range(self.horizons):
#                    actions_taken.append(np.array([action_to_take]))
#                if(self.actions_ag=='nc'):
#                    actions_taken.append(np.array([clean_action]))
#                if(self.actions_ag=='cc'):
#                    actions_taken.append(np.array([clean_action]))

                #this is the observation returned by taking a step in the env
#            obs=np.copy(next_state)

                #get the next state (usable by NN)
#            just_one=True
#                next_state = from_observation_to_usablestate(next_state, self.which_agent, just_one)
#            next_state = obs                   ##?############changed by sun
            curr_state=np.copy(next_state)
            for i in range (self.horizons):
                traj_taken.append(next_states[i])


#            print ("###nextstate###",next_state[1].tolist(),"####nextstate###")
                #bookkeeping
            if(not(self.print_minimal)):
                if((step+self.horizons)%100==0):
                    print("done step ", step+self.horizons, ", rew: ", total_reward_for_episode)
            step+=self.horizons

                #when to stop  #####(step>=self.steps_per_episode) or (
            if(follow_trajectories):
                    if((step>=self.steps_per_episode) or (curr_line_segment>5800)): #6289,5800
                        stop_taking_steps = True
#            else:
#                    if(step>=self.steps_per_episode):
#                        stop_taking_steps = True
            
#        if(not(self.print_minimal)):
#            print("DONE TAKING ", step, " STEPS.")
#            print("Reward: ", total_reward_for_episode)

#        mydict = dict(
#        observations=tensor_utils.stack_tensor_list(observations),
#        actions=tensor_utils.stack_tensor_list(actions_taken))
#        rewards=tensor_utils.stack_tensor_list(rewards),
#        agent_infos=agent_infos,
#        env_infos=tensor_utils.stack_tensor_dict_list(env_infos))
        print (step)
        return accumulate_reward,list_accumulate_reward,traj_taken, actions_taken ##, mydict  #total_reward_for_episode,

    def get_action(self, curr_nn_state, curr_line_segment, reward_func):

    # ###curr_nn_state (1,8),xy,fi,u,v,r,sin,cos


        #randomly sample N candidate action sequences
###        all_samples_Tu = npr.uniform(-2,2, (self.N, self.horizon, 1))
###        all_samples_Tr = npr.uniform(-1.5,1.5, (self.N, self.horizon, 1))
###        all_samples = np.concatenate((all_samples_Tu,all_samples_Tr),axis=2)       #####(N,horizon,2)

####################################重复第一组序列，#######################
        all_samples_Tu = npr.uniform(0, 20, (self.N, 1 , 1))
        all_samples_Tr = npr.uniform(-10,10, (self.N, 1, 1))
        all_samples = np.concatenate((all_samples_Tu,all_samples_Tr),axis=2)       #####(N,1,2)
        all_samples = np.tile(all_samples, (1, self.horizon, 1))     ####(N,H,2) ，（5000,20，2）

        #forward simulate the action sequences (in parallel) to get resulting (predicted) trajectories
        many_in_parallel = True
        ####get next states x,y,fi,u,v,r

        resulting_states = self.dyn_model.do_forward_sim([curr_nn_state,0], np.copy(all_samples), many_in_parallel)
        resulting_states = np.array(resulting_states) #this is [horizon+1, N, statesize] = [21,5000,8]
#
    #  print ("####resulting_state####",resulting_states.shape)
        #init vars to evaluate the trajectories
        scores=np.zeros((self.N,))
        done_forever=np.zeros((self.N,))
        move_to_next=np.zeros((self.N,))
        curr_seg = np.tile(curr_line_segment,(self.N,))
        curr_seg = curr_seg.astype(int)
        prev_forward = np.zeros((self.N,))
        moved_to_next = np.zeros((self.N,))
        prev_pt = resulting_states[0]  # prev_pt是3维数组的第一个，(5000,8)，2维数组，初始状态
#        print ("####pt_number###",resulting_states.shape[0])
        #accumulate reward over each timestep

        accumulate_reward = 0
        for pt_number in range(resulting_states.shape[0]):    ###resulting_states.shape[0]=H+1，21
#            print (pt_number,resulting_states.shape[0])
            #array of "the point"... for each sim
            pt = resulting_states[pt_number] # N x state，（5000,8）

            #how far is the point from the desired trajectory
            #how far along the desired traj have you moved since the last point
##3-13            min_perp_dist, curr_forward, curr_seg, moved_to_next = self.calculate_geometric_trajfollow_quantities(pt, curr_seg+pt_number, moved_to_next)

            #update reward score
##3-13            scores, done_forever = reward_func(pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, curr_seg,
##3-13                                                moved_to_next, done_forever, all_samples, pt_number)
            scores, done_forever = reward_func(pt, prev_pt, scores, curr_seg + pt_number, done_forever, all_samples, pt_number)

###         print (pt_number)###  0---5
            #update vars
##3-13            prev_forward = np.copy(curr_forward)
            prev_pt = np.copy(pt)


            accumulate_reward+= scores

        print("最终的累计奖励为：",accumulate_reward)

        #np.save(save_dir + '/control_iter' + str(counter_agg_iters) + '.npy', selected_u)

         #pick best action sequence
        best_score = np.min(scores)
        best_sim_number = np.argmin(scores)
        best_sequence = all_samples[best_sim_number]
        best_action = np.copy(best_sequence[0])###just one action
        best_curr_seg = curr_seg[best_sim_number]
        print ("#######bestscore#########",best_score)
        

        return best_score ,best_action, best_sim_number, best_sequence, moved_to_next ,best_curr_seg

    def calculate_geometric_trajfollow_quantities(self, pt, curr_seg, moved_to_next):

        #arrays of line segment points... for each sim
        curr_start = self.desired_states[curr_seg]
        curr_end = self.desired_states[curr_seg+1]
        next_start = self.desired_states[curr_seg+1]
        next_end = self.desired_states[curr_seg+2]
#        print("curr_statr#######",self.desired_states.shape,curr_start.shape,curr_start,self.desired_states,curr_seg.shape)
        #initialize
        min_perp_dist = np.ones((self.N, ))*5000

        ####################################### closest distance from point to current line segment
        #### pt =[21,states][21,hang,yi ge rollout]
        #vars
        a = pt[:,self.x_index]- curr_start[:,0] 
        b = pt[:,self.y_index]- curr_start[:,1]
        c = curr_end[:,0]- curr_start[:,0]
        d = curr_end[:,1]- curr_start[:,1]
#        print(a.shape)
        #####compared (a,c)and (b,d),equal is perfected.

        #project point onto line segment
        which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), (np.multiply(c,c) + np.multiply(d,d)))

        #point on line segment that's closest to the pt
        closest_pt_x = np.copy(which_line_section)
        closest_pt_y = np.copy(which_line_section)
        closest_pt_x[which_line_section<0] = curr_start[:,0][which_line_section<0]
        closest_pt_y[which_line_section<0] = curr_start[:,1][which_line_section<0]
        closest_pt_x[which_line_section>1] = curr_end[:,0][which_line_section>1]
        closest_pt_y[which_line_section>1] = curr_end[:,1][which_line_section>1]
        closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,0] + 
                            np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
        closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (curr_start[:,1] + 
                            np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

        #min dist from pt to that closest point (ie closes dist from pt to line segment)
        min_perp_dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + 
                                (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

        ####################################### "forward-ness" of the pt... for each sim
        curr_forward = which_line_section

        ###################################### closest distance from point to next line segment

        #vars
        a = pt[:,self.x_index]- next_start[:,0]
        b = pt[:,self.y_index]- next_start[:,1]
        c = next_end[:,0]- next_start[:,0]
        d = next_end[:,1]- next_start[:,1]

        #project point onto line segment
        which_line_section = np.divide((np.multiply(a,c) + np.multiply(b,d)), 
                                        (np.multiply(c,c) + np.multiply(d,d)))

        #point on line segment that's closest to the pt
        closest_pt_x = np.copy(which_line_section)
        closest_pt_y = np.copy(which_line_section)
        closest_pt_x[which_line_section<0] = next_start[:,0][which_line_section<0]
        closest_pt_y[which_line_section<0] = next_start[:,1][which_line_section<0]
        closest_pt_x[which_line_section>1] = next_end[:,0][which_line_section>1]
        closest_pt_y[which_line_section>1] = next_end[:,1][which_line_section>1]
        closest_pt_x[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,0] + 
                            np.multiply(which_line_section,c))[np.logical_and(which_line_section<=1, which_line_section>=0)]
        closest_pt_y[np.logical_and(which_line_section<=1, which_line_section>=0)] = (next_start[:,1] + 
                            np.multiply(which_line_section,d))[np.logical_and(which_line_section<=1, which_line_section>=0)]

        #min dist from pt to that closest point (ie closes dist from pt to line segment)
        dist = np.sqrt((pt[:,self.x_index]-closest_pt_x)*(pt[:,self.x_index]-closest_pt_x) + 
                        (pt[:,self.y_index]-closest_pt_y)*(pt[:,self.y_index]-closest_pt_y))

        ############################################ 

        #pick which line segment it's closest to, and update vars accordingly
        curr_seg[dist<=min_perp_dist] += 1
        moved_to_next[dist<=min_perp_dist] = 1
        curr_forward[dist<=min_perp_dist] = which_line_section[dist<=min_perp_dist]
        min_perp_dist = np.min([min_perp_dist, dist], axis=0)

        return min_perp_dist, curr_forward, curr_seg, moved_to_next
