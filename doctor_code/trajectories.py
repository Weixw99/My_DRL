import numpy as np
import math
import time

def make_trajectory(shape, starting_state):

    curr_x = np.copy(starting_state[0])
    curr_y = np.copy(starting_state[1])
    angle = np.copy(starting_state[2])
#    curr_x =  x_index    ###changed by sun   x_postion
#    curr_y =  y_index    ###changed by sun   y_postion

    my_list = []

    if(shape=="left_turn"):
        if(1==1):
            my_list.append(np.array([curr_x, curr_y]))
            my_list.append(np.array([curr_x+2, curr_y]))
            my_list.append(np.array([curr_x+4, curr_y]))
            my_list.append(np.array([curr_x+6, curr_y]))
            my_list.append(np.array([curr_x+6, curr_y+2]))
            my_list.append(np.array([curr_x+6, curr_y+3]))
            my_list.append(np.array([curr_x+6, curr_y+4]))
            my_list.append(np.array([curr_x+6, curr_y+5]))
            my_list.append(np.array([curr_x+6, curr_y+6]))
            my_list.append(np.array([curr_x+6, curr_y+7]))
#        else:
#            my_list.append(np.array([curr_x, curr_y]))
#            my_list.append(np.array([curr_x+1, curr_y]))
#            my_list.append(np.array([curr_x+2, curr_y]))
#            my_list.append(np.array([curr_x+3, curr_y]))
#            my_list.append(np.array([curr_x+4, curr_y+1]))
#            my_list.append(np.array([curr_x+4, curr_y+2]))
#            my_list.append(np.array([curr_x+4, curr_y+3]))
#            my_list.append(np.array([curr_x+4, curr_y+4]))

    if(shape=="sin"):
        i=0
        a=0.3
        while (i <= 4*math.pi):
            my_list.append(np.array([curr_x + i, curr_y + np.sin(i)]))  ###0-1.8,0
            i += a

#        else:
#            my_list.append(np.array([curr_x, curr_y]))
#            my_list.append(np.array([curr_x, curr_y+1]))
#            my_list.append(np.array([curr_x, curr_y+2]))
#            my_list.append(np.array([curr_x+2, curr_y+3]))
#            my_list.append(np.array([curr_x+3, curr_y+3]))
#            my_list.append(np.array([curr_x+4, curr_y+3]))
#            my_list.append(np.array([curr_x+5, curr_y+3]))
#            my_list.append(np.array([curr_x+6, curr_y+3]))
#            my_list.append(np.array([curr_x+7, curr_y+3]))
#            my_list.append(np.array([curr_x+8, curr_y+3]))

    if(shape=="u_turn"):   ###31
        a = 0.45
        b = 0.45
        c = 0.45
        i = 0
        j = 0
#        while (i <= 1.8):
#            my_list.append(np.array([curr_x + i, curr_y]))
#            i += a
#        my_list.append(np.array([curr_x + 2.1, curr_y+0.25]))
#        my_list.append(np.array([curr_x + 2.4, curr_y + 0.5]))
#        my_list.append(np.array([curr_x + 2.7, curr_y + 0.75]))
#        while (i <= 10):
#            my_list.append(np.array([curr_x + 3, curr_y+i-0.8]))
#            i += 0.45
#        while (i < 11.8):
#            my_list.append(np.array([curr_x + 3, curr_y+]))
#            i += 0.45
        while(i <=  100*a+0.000001):
            my_list.append(np.array([curr_x+i, curr_y])) ###0,0-50,i=1.8+a                          1右横
            i+=a
#       my_list.append(np.array([curr_x+2.1, curr_y+0.25]))
#       my_list.append(np.array([curr_x+2.4, curr_y+0.5]))
#       my_list.append(np.array([curr_x+2.7, curr_y+0.75]))
###################################################################
        while(j <= 40*b+0.000001):
            my_list.append(np.array([curr_x+100*a+2.0, curr_y+j+2.0]))###5,0-10                     2上
            j+=b
#       my_list.append(np.array([curr_x+3.3, curr_y+11.15]))
#       my_list.append(np.array([curr_x+3.6, curr_y+11.4]))
#       my_list.append(np.array([curr_x+3.9, curr_y+11.65]))
        j=41*b
###################################################################
        while(i <= 197*a+0.000001):
            my_list.append(np.array([curr_x+(100*a-2*a)+101*a-i, curr_y+41*b+4.05]))###25,0-10          3左横
            i+=a   
#       my_list.append(np.array([curr_x+6.3, curr_y+11.65]))
#       my_list.append(np.array([curr_x+6.6, curr_y+11.4]))
#       my_list.append(np.array([curr_x+6.9, curr_y+11.15]))
####################################print (my_list[len(my_list)-1])
        while(j <= 81*b+0.000001):
            my_list.append(np.array([curr_x, curr_y+j+4*a+10*b-0.9]))###5,0-10                     4上
            j+=b
#       my_list.append(np.array([curr_x+7.5, curr_y+0.75]))
#       my_list.append(np.array([curr_x+7.8, curr_y+0.5]))
#       my_list.append(np.array([curr_x+8.1, curr_y+0.25]))
######################################################################
        while(i <= 296*a+0.000001):
            my_list.append(np.array([curr_x+i-198*a+2*b, curr_y+82*b+5*a+10*b+0.1]))###25,0-10     5右横
            i+=a
#       my_list.append(np.array([curr_x+7.5, curr_y+0.75]))
#       my_list.append(np.array([curr_x+7.8, curr_y+0.5]))
#       my_list.append(np.array([curr_x+8.1, curr_y+0.25]))
######################################################################
        while(j <= 122*b+0.000001):
            my_list.append(np.array([curr_x+100*a+2.0, curr_y+j+5*a+10*b+2.1]))###5,0-10            6上
            j+=b
#       my_list.append(np.array([curr_x+7.5, curr_y+0.75]))
#       my_list.append(np.array([curr_x+7.8, curr_y+0.5]))
#       my_list.append(np.array([curr_x+8.1, curr_y+0.25]))
######################################################################
        while(i <= 394*a+0.000001):
            my_list.append(np.array([curr_x+(100*a-2*a)+297*a-i, curr_y+123*b+5*a+10*b+3.8]))###25,0-10  7左横
            i+=a   


    if(shape=="straight"):
        i=0
        num_pts = 8000
        while(i < num_pts):
            my_list.append(np.array([curr_x+i, curr_y]))
            i+=0.2

    if(shape==" "):
        i=0
        num_pts = 40
        while(i < num_pts):
            my_list.append(np.array([curr_x-i, curr_y]))
            i+=0.5

    if(shape=="forward_backward"):
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+1, curr_y]))
        my_list.append(np.array([curr_x+2, curr_y]))
        my_list.append(np.array([curr_x+3, curr_y]))
        my_list.append(np.array([curr_x+2, curr_y]))
        my_list.append(np.array([curr_x+1, curr_y]))
        my_list.append(np.array([curr_x+0, curr_y]))
        my_list.append(np.array([curr_x-1, curr_y]))
        my_list.append(np.array([curr_x-2, curr_y]))

    if(shape=="circle"):
        num_pts = 6500        ####6290可以完成一整圈
        radius= 60            # 30
        speed= 0.0013         #0.0013
        for i in range(num_pts):
#            curr_x= radius*np.cos(speed*i-math.pi) #- radius
#            curr_y= radius*np.sin(speed*i-math.pi)
#            curr_x = radius * np.cos(speed * i - 1.5 * math.pi)  # - radius
#            curr_y = radius * np.sin(speed * i - 1.5 * math.pi)
            curr_x = radius * np.cos(speed * i )  # - radius
            curr_y = radius * np.sin(speed * i )
            angle = i * speed + 0.5 * np.pi
#            curr_x = radius*0 + (speed * i )  # - radius
#            curr_y = radius*0 + (speed * i )
#            angle =  0.25 * np.pi
            my_list.append(np.array([curr_x, curr_y, angle]))
            #time.sleep(0.1)
#    print ("###mylist##",np.array(my_list),np.array(my_list).shape)
    return np.array(my_list)

def get_trajfollow_params(desired_traj_type):

#    desired_snake_headingInit= 0
    horiz_penalty_factor= 0
    forward_encouragement_factor= 0
    heading_penalty_factor= 0
    r_penalty_factor= 0
    u_penalty_factor= 0
    if(1==1):
        if(desired_traj_type=="sin"):
            horiz_penalty_factor= 310
            forward_encouragement_factor= 470
            heading_penalty_factor= 1

        if(desired_traj_type=="circle"):
            horiz_penalty_factor= 100                    ####  x  182
            forward_encouragement_factor= 100            ####  y  182
            heading_penalty_factor= 0.1                  ####  angle  5
            r_penalty_factor=  0                         ####  r      5.5
            u_penalty_factor= 0.0                        ####  u      0

        if(desired_traj_type=="straight"):
            horiz_penalty_factor= 400
            forward_encouragement_factor= 560
            heading_penalty_factor= 6.2

        if(desired_traj_type=="u_turn"):
            horiz_penalty_factor=680 ###330  400          530   680
            forward_encouragement_factor= 327  ###470 560 560   560  700 590
            heading_penalty_factor= 6.6 ###1.5  5        10    9.5

    return horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, r_penalty_factor, u_penalty_factor  #, desired_snake_headingInit
