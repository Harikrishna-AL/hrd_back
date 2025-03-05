import math

import numpy as np
from gym import utils

from thermal_regulation.envs.thermal_regulation_env import ThermalRegulationEnv
from thermal_regulation.envs.mymujoco import MyMujocoEnv


class MySwimmerEnv(MyMujocoEnv, utils.EzPickle):
    FILE = "swimmer.xml"
    IS_WALKER = False

    def __init__(self, xml_path, *args, **kwargs):
        MyMujocoEnv.__init__(self, xml_path, 50)  # TODO: check frame skip from file
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self.get_current_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def get_current_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self.get_current_obs()


class SwimmerThermalRegulationEnv(ThermalRegulationEnv):
    """
    Two Resource Problem with approximated scale of Konidaris & Barto paper
    """
    MODEL_CLASS = MySwimmerEnv
    ORI_IND = 2


class SwimmerSmallThermalRegulationEnv(ThermalRegulationEnv):
    """
    Small-sized Two Resource Problem with the scale of original GatherEnv
    """
    MODEL_CLASS = MySwimmerEnv
    ORI_IND = 2

    def __init__(self,
                 activity_range=6.,
                 n_bins=20,
                 sensor_range=12.,
                 *args, **kwargs):
        super().__init__(
            n_bins=n_bins,
            activity_range=activity_range,
            sensor_range=sensor_range,
            *args, **kwargs
        )
