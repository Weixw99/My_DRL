
import numpy as np

starting_state_NN= [1, 11, 6,11,15,19]

curr_x = np.copy(starting_state_NN[2])
curr_y = np.copy(starting_state_NN[3])
my_list=[]

if(1==1):
    i=0
    num_pts = 5
    while(i < num_pts):
        my_list.append(np.array([curr_x+i, curr_y]))
        i+=1


my_list=np.array(my_list)
print(my_list)
print(my_list.shape)  # (5,2)

