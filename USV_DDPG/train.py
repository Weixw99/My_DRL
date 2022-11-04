import sys
import os

import numpy as np

from DDPG.env import OUNoise
from common.utils import action_obs

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径


def train(cfg, env, agent, plot_cfg):
    print('开始训练！')
    print(f'环境：{cfg.env_name}，算法：{cfg.algo_name}，设备：{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        action_num = []
        while not done:
            env.render()
            i_step += 1
            action = agent.choose_action(state)
            action_num.append(action)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if (i_step+1) % cfg.ep_num == 0:
                done = True
        print('回合：{}/{}，奖励：{:.2f}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if i_ep % 2000 == 0 and i_ep != 0:
            agent.save_process(path=(plot_cfg.model_path + str(i_ep) + '.pt'))
    print('完成训练！')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset() 
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            env.render()
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if (i_step+1) % cfg.ep_num == 0:
                done = True
        print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.test_eps, ep_reward))
        rewards.append(ep_reward)

        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成测试！')
    return rewards, ma_rewards
