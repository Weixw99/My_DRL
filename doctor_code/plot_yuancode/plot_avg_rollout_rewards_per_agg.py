
import numpy as np

avg_rollout_rewards_per_agg = np.load('E:/pycharm2018/project/circle_0326_yanzheng/run_0/avg_rollout_rewards_per_agg.npy')
#list_old_loss = np.load('E:/pycharm2018/project/circle_0326_yanzheng/run_0/losses/list_old_loss.npy')

#list_training_loss = np.load('E:/pycharm2018/project/circle_0326_yanzheng/run_0/losses/list_training_loss.npy')

print("avg_rollout_rewards_per_agg为：",avg_rollout_rewards_per_agg)  # [ 0.          0.01400006  0.01850381  0.02117461  0.02237916  0.02117025]
print("avg_rollout_rewards_per_agg.shape为：",avg_rollout_rewards_per_agg.shape)  # (6,)


