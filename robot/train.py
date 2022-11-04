import sys
import os
import time
import numpy as np

from env import OUNoise
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
        while not done:
            if cfg.display:
                env.render()
            else:
                if cfg.train_eps - i_ep < 100:
                    env.render()
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if (i_step+1) % cfg.ep_num == 0:
                done = True
        print('回合：{}/{}，奖励：{:.2f}'.format(i_ep+1, cfg.train_eps, ep_reward))
        if (i_ep + 1) % 500 == 0:
            agent.save_process(path=(plot_cfg.model_path + 'train{}.pt'.format(i_ep + 1)))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
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
            time.sleep(0.3)
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


# 更改core最下面和order_enforcing
def simulation(cfg, env, agent):
    from DRL_Client import TCPClient
    print('开始仿真！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    wxw = TCPClient('172.22.0.47', 1231)
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    state = env.reset()
    done = False
    ep_reward = 0
    i_step = 0
    while not done:
        env.render()
        i_step += 1
        rev_data = wxw.get_data()  # 等待接收数据,收到处理好返回
        state[0], state[1] = rev_data[1], rev_data[0]  # 更新位置信息
        action = agent.choose_action(state)
        send_aim = [action[1] + state[1], action[0] + state[0]]
        send_aim = list(map(float, send_aim))
        wxw.send_data(send_aim)  # 将下一步的目标点发送出去
        next_state, reward, done, _ = env.step(action, state)
        print('state:', state)
        print('aim:', send_aim)
        ep_reward += reward
        state = next_state  # 此时state是当前船的位置加上采取的动作之后的位置
    rewards.append('奖励：{}'.format(ep_reward))
    print('完成仿真！')
    return rewards, ma_rewards
