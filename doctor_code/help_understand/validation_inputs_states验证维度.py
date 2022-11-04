import numpy as np

a = [1, 2] ; b = [4, 5]
c = [7, 8];d = [11,15];e = [20,30];f = [0,0]
k = [33,44];l = [99,77];m = [11,22]
states_val = np.array([[a,b,c],[d,e,f],[k,l,m]])

print("states_val输出为:")
print(states_val)
print("******************************")

validation_inputs_states = []
length_curr_rollout = 5
for i in range(2):
    validation_inputs_states.append(states_val[i][0:length_curr_rollout - 3])

    print("validation_inputs_states为：")
    print(validation_inputs_states)
    print('\n')

validation_inputs_states = np.concatenate(validation_inputs_states)
print(validation_inputs_states) # 现在是2维数组



