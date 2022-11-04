from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'

    def __init__(self, *args, **kwargs):
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
            self.get_body_comvel("torso"),
        ]).reshape(-1)

    def step(self, action, collectingInitialData=False):
        xposbefore = self.get_body_com("torso")[0]
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        xposafter = self.get_body_com("torso")[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0
        survive_reward = 0.05

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self._state
        notdone = np.isfinite(state).all() \
            and self.get_body_com("torso")[2] >= 0.3 and self.get_body_com("torso")[2] <= 1.0 #used to be 0.2, state[2]
        done = not notdone
        ob = self.get_current_obs()

        return Step(ob, float(reward), done)

    def get_my_sim_state(self):
        my_sim_state=np.squeeze(np.concatenate((self.model.data.qpos, self.model.data.qvel, self.model.data.qacc, self.model.data.ctrl)))
        return my_sim_state

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

