import numpy as np

a = [1, 2] ; b = [4, 5]
c = [7, 8];d = [11,15];e = [20,30];f = [0,0]
k = [33,44];l = [99,77];m = [11,22]
w1 = np.array([[a,b,c],[d,e,f],[k,l,m]])

print("w1输出为")
print(w1)

differences=[]
for states_in_single_rollout in w1:
    output = states_in_single_rollout[1:states_in_single_rollout.shape[0], :] \
             - states_in_single_rollout[0:states_in_single_rollout.shape[0] - 1, :]

    differences.append(output)

print("differences为：",differences)

output = np.concatenate(differences, axis=0)

print("最终的output为：",output)















