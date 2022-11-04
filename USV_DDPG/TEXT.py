import os
import sys
import argparse

from MAenvs.multiagent.environment import MultiAgentEnv
from MAenvs.multiagent.policy import InteractivePolicy
import MAenvs.multiagent.scenarios as scenarios

sys.path.insert(1, os.path.join(sys.path[0], '..'))

if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # 从脚本中加载情景
    scenario = scenarios.load(args.scenario).Scenario()
    # create multiagent environment
    env = MultiAgentEnv(scenario.make_world(), scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None, shared_viewer=False)
    # 渲染调用以创建查看器窗口（仅对交互式政策有必要）。
    env.render()
    # 为每个agent创建互动政策
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # 执行循环
    obs_n = env.reset()
    while True:
        # 从每个代理人的政策中查询行动
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # 呈现所有代理视图
        env.render()
        # display rewards
        for agent in env.world.agents:
            print(agent.name + " reward: %0.3f" % env._get_reward(agent))
