

import numpy as np

a = [1, 2,22,1,4] ; b = [4, 5,55,2,6]
c = [7, 8,88,5,7];d = [11,15,155,3,1];e = [20,30,300,99,34];f = [0,21,96,23,22]
k = [33,44,55,993,998];l = [99,77,65,899,778];m = [11,22,23,699,567];n=[100,101,234,999,700];p=[102,103,900,0,57];q=[104,105,600,44,55]
states_val = np.array([[a,b,c,d],[e,f,k,l],[m,n,p,q]])

print("states_val输出为")
print(states_val)  # (3,4,5)
print("states_val.shape为：",states_val.shape)  # (3,4,5)

a1 = [1, 2] ; b1 = [4, 5]
c1 = [7, 8];d1 = [11,15];e1 = [20,30];f1 = [0,21]
k1 = [33,44];l1 = [99,77];m1 = [11,22];n1=[100,101];p1=[102,103];q1=[104,105]
controls_val = np.array([[a1,b1,c1,d1],[e1,f1,k1,l1],[m1,n1,p1,q1]])

print("control_val输出为")
print(controls_val)  # (3,4,2)
print("controls_val.shape为：",controls_val.shape)  # (3,4,2)


labels_1step = []
controls_100step=[]
validation_inputs_states=[]

for i in range(3): # i取0,1,2

    length_curr_rollout = 4
    if (length_curr_rollout > 2):

        validation_inputs_states.append(states_val[i][0:length_curr_rollout - 2])
        print("列表validation_inputs_states为：", validation_inputs_states)

        list_100 = []
        for j in range(3):
            list_100.append(controls_val[i][0 + j:length_curr_rollout - 2 + j])

        print("j的循环后list_100为：",list_100)


        # 下面输出的这个list_100是3维数组
        list_100 = np.array(list_100)  # ，3维数组，

        print("数组list_100为：", list_100)  # (3,2,2),中间的2指的是length_curr_rollout - 2
        print("数组list_100.shape为：", list_100.shape)

        # np.swapaxes(list_100,0,1)，指3维数组list_100的轴0与轴1进行交换，
        # 查看help_understand里的np.swapaxes.py
        # 交换轴是为了100个控制？？
        list_100 = np.swapaxes(list_100, 0, 1)  # (2,3,2)
        print("交换数轴后，数组list_100.shape为：", list_100.shape)
        #
        controls_100step.append(list_100)   # list_100每次循环清空，但是controls_100step没清空
        print("controls_100step为：", controls_100step)



        labels_1step.append(states_val[i][0 + 1:length_curr_rollout - 2 + 1])
        print("labels_1step为：", labels_1step)

validation_inputs_states = np.concatenate(validation_inputs_states)  # (20*233,16)
print("validation_inputs_states为：", validation_inputs_states)

print("validation_inputs_states.shape为：", validation_inputs_states.shape)  # (6,3)
# # 这个的理解在ccontrols_100step.py中，输出的controls_100step是3维，
#
#
controls_100step = np.concatenate(controls_100step)
print("controls_100step.shape为：", controls_100step.shape) #(6, 2, 2)

# labels_1step是2维数组，下面几个也是
labels_1step = np.concatenate(labels_1step)  # 2维数组，

print("labels_1step.shape为：", labels_1step.shape)  # (6, 3)







