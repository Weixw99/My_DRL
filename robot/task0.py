import sys
import os
import datetime
import gym
import torch
from env import NormalizedActions
from agent import DDPG
from train import train, test, simulation
from common.utils import save_results, make_dir
from common.utils import plot_rewards

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DDPGConfig:
    def __init__(self):
        self.type = 'train'  # 运行模式   train  test  simulation
        self.display = False
        self.algo_name = 'DDPG'  # 算法名称
        self.env_name = 'aim-usv-sim-v0' if self.type == 'simulation' else 'aim-usv-v0'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 30000  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        self.gamma = 0.99  # 折扣因子
        self.critic_lr = 1e-2  # 评论家网络的学习率
        self.actor_lr = 1e-3  # 演员网络的学习率
        self.memory_capacity = 8000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim1 = 64  # 网络隐藏层维度
        self.hidden_dim2 = 64  # 网络隐藏层维度
        self.soft_tau = 1e-3  # 软更新参数
        self.ep_num = 500


class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = 'DDPG'  # 算法名称
        self.env_name = "aim-usv-v0"  # 环境名称
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.test_path = curr_path + "/outputs/" + self.env_name + '/20220613-205847/'
        self.sim_path = curr_path + '/outputs/aim-usv-v0/20220613-205847/'
        self.save = True  # 是否保存图片
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU


def env_agent_config(cfgs, seed=1):
    envs = NormalizedActions(gym.make(cfgs.env_name))  # 装饰action噪声
    envs.seed(seed)  # 随机种子
    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    agents = DDPG(state_dim, action_dim, cfgs)
    return envs, agents


cfg = DDPGConfig()
plot_cfg = PlotConfig()
if cfg.type == 'train':
    # 训练
    env, agent = env_agent_config(cfg, seed=1)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    rewards, ma_rewards = train(cfg, env, agent, plot_cfg)
    agent.save(path=plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果

elif cfg.type == 'test':
    # 测试
    env, agent = env_agent_config(cfg, seed=10)
    agent.load_process(path=(plot_cfg.test_path + 'models/train3000.pt'))
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test', path=(plot_cfg.test_path + 'results/'))
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果

elif cfg.type == 'simulation':
    # Simulation
    env, agent = env_agent_config(cfg, seed=10)
    agent.load_process(path=(plot_cfg.sim_path + 'models/train2000.pt'))
    rewards, ma_rewards = simulation(cfg, env, agent)
