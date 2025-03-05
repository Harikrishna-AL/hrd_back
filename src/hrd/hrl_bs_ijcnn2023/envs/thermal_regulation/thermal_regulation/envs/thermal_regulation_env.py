import math
import os
import tempfile
import xml.etree.ElementTree as ET
import inspect
from enum import Enum, auto
from copy import copy
from logging import warning

import glfw
import numpy as np
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv, DEFAULT_SIZE
from gym import utils
from mujoco_py.generated import const

from thermal_regulation.envs.thermal_module import ThermalModule, Climate

BIG = 1e6
DEFAULT_CAMERA_CONFIG = {}
TEMP_DIFF = (42. - 38.) / 6.0


class ObjectClass(Enum):
    FOOD = auto()


class InteroClass(Enum):
    ENERGY = auto()
    TEMPERATURE = auto()


def qtoeuler(q):
    """ quaternion to Euler angle

    :param q: quaternion
    :return:
    """
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


class ThermalRegulationEnv(MujocoEnv, utils.EzPickle):
    MODEL_CLASS = None
    ORI_IND = None

    def __init__(self,
                 ego_obs=True,
                 n_food=6,
                 activity_range=42,
                 robot_object_spacing=2.,
                 catch_range=1.,
                 n_bins=10,
                 sensor_range=12,
                 sensor_span=2 * math.pi,
                 coef_inner_rew=0.,
                 coef_main_rew=100.,
                 coef_ctrl_cost=0.001,
                 coef_head_angle=0.005,
                 dying_cost=-10,
                 max_episode_steps=np.inf,
                 show_sensor_range=False,
                 reward_setting="homeostatic_shaped",
                 reward_bias=None,
                 internal_reset="random",
                 energy_random_range=(-1 / 6., 1 / 6.),
                 temperature_random_range=(38. - TEMP_DIFF, 38. + TEMP_DIFF),
                 visualize_temp=False,
                 random_climate=False,
                 *args, **kwargs):
        """

        :param int n_food:  Number of greens in each episode
        :param float activity_range: he span for generating objects (x, y in [-range, range])
        :param float robot_object_spacing: Number of objects in each episode
        :param float catch_range: Minimum distance range to catch an food
        :param float shade_range: Maximum distance range to be inside the shade
        :param int n_bins: Number of objects in each episode
        :param float sensor_range: Maximum sensor range (how far it can go)
        :param float sensor_span: Maximum sensor span (how wide it can span), in radians
        :param coef_inner_rew:
        :param coef_main_rew:
        :param coef_cost:
        :param coef_head_angle:
        :param dying_cost:
        :param max_episode_steps:
        :param show_sensor_range: Show range sensor. Default OFF
        :param reward_setting: Setting of the reward definitions. "homeostatic", "homeostatic_shaped", "one", "homeostatic_biased" or "greedy". "homeostatic_shaped" is default. "greedy is not a homeostatic setting"
        :param reward_bias: biasing reward with constant. new_reward = reward + reward_bias
        :param internal_reset: resetting rule of the internal nutrient state. "setpoint" or "random".
        :param energy_random_range: if reset condition is "random", use this region for initialize energy variable
        :param temperature_random_range: if reset condition is "random", use this region for initialize temperature variable (in Celsius degree)
        :param visualize_temp: whether visualize the temperature on the body or not
        :param random_climate: whether randomly climate changes (sunny or cloudy)
        :param args:
        :param kwargs:
        """
        self.n_food = n_food
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.coef_main_rew = coef_main_rew
        self.coef_ctrl_cost = coef_ctrl_cost
        self.coef_head_angle = coef_head_angle
        self.dying_cost = dying_cost
        self._max_episode_steps = max_episode_steps
        self.show_sensor_range = show_sensor_range
        self.reward_setting = reward_setting
        self.reward_bias = reward_bias if reward_bias else 0.
        self.internal_reset = internal_reset
        self.energy_random_range = energy_random_range
        self.temperature_random_range = temperature_random_range
        self.visualize_temp = visualize_temp
        self.random_climate = random_climate

        self.objects = []
        self.viewer = None

        # Internal state
        self._target_internal_state = np.array([0.0, 0.0])  # [Blue, Red]

        if self.internal_reset in {"setpoint", "random"}:
            self.internal_state = {
                InteroClass.ENERGY: 0.0,
                InteroClass.TEMPERATURE: 0.0,
            }
        else:
            raise ValueError

        self.prev_interoception = self.get_interoception()
        self.success_consumptive_act = 0.1
        self.default_metabolic_update = 0.00015
        self.survival_area = 1.0

        # Thermal Dynamics Parameters
        # Temperature configuration in Celsius degree
        self.temp_earth = 30 # 40.
        self.temp_air = 25 # 35.
        self.target_temperature = 38.

        # Viability range of the robot
        self.temp_limit_max = 42.
        self.temp_limit_min = 34.

        self.thermal_model = ThermalModule()  # note: all methods of this model use temperature in Celsius degree
        self._climate = Climate.default

        utils.EzPickle.__init__(**locals())

        # for openai baseline
        self.reward_range = (-float('inf'), float('inf'))
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        MODEL_DIR = os.path.join(p.parent, "models", model_cls.FILE)

        tree = ET.parse(MODEL_DIR)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 1" % walldist,
                size="%d.5 0.5 2" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 1" % walldist,
                size="%d.5 0.5 2" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 1" % walldist,
                size="0.5 %d.5 2" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 1" % walldist,
                size="0.5 %d.5 2" % walldist))

        asset = tree.find(".//asset")
        ET.SubElement(
            asset, "hfield", dict(
                name="hill",
                file=os.path.join(p.parent, "models", "hill_height.png"),
                size="10 10 0.25 0.1"
            ))

        with tempfile.NamedTemporaryFile(mode='wt', suffix=".xml") as tmpfile:
            file_path = tmpfile.name
            tree.write(file_path)

            # build mujoco
            self.wrapped_env = model_cls(file_path, **kwargs)

        # optimization, caching obs spaces
        ub = BIG * np.ones(self.get_current_obs().shape, dtype=np.float32)
        self.obs_space = spaces.Box(ub * -1, ub)
        ub = BIG * np.ones(self.get_current_robot_obs().shape, dtype=np.float32)
        self.robot_obs_space = spaces.Box(ub * -1, ub)

        ub = np.ones(len(self.wrapped_env.action_space.high) + 1, dtype=np.float32)
        self.act_space = spaces.Box(ub * -1, ub)

        self.max_episode_length = self._max_episode_steps

        self._step = 0

        self.num_food_eaten = 0

        self.leaf_height = 1
        self.leaf_height_var = 2

    @property
    def dim_intero(self):
        return np.prod(self._target_internal_state.shape)

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_internal_state(self):
        if self.internal_reset == "setpoint":

            self.thermal_model.reset(self.target_temperature)

            self.internal_state = {
                InteroClass.ENERGY: 0.0,
                InteroClass.TEMPERATURE: self.scale_temperature(self.target_temperature),
            }

        elif self.internal_reset == "random":

            temp = self.wrapped_env.np_random.uniform(self.temperature_random_range[0],
                                                      self.temperature_random_range[1])

            self.thermal_model.reset(temp)

            self.internal_state = {
                InteroClass.ENERGY: self.wrapped_env.np_random.uniform(self.energy_random_range[0],
                                                                       self.energy_random_range[1]),
                InteroClass.TEMPERATURE: self.scale_temperature(temp),
            }

        else:
            raise ValueError

    def reset(self, climate=None, n_food=None):
        self._step = 0

        if n_food is not None:
            self.n_food = n_food

        self.num_food_eaten = 0

        if self.wrapped_env.np_random is None:
            self.wrapped_env.seed()
        self.wrapped_env.reset()

        if climate is not None:
            self._climate = climate
        elif self.random_climate:
            self._climate = self.wrapped_env.np_random.choice(a=(Climate.sunny, Climate.cloudy))

        self.reset_internal_state()

        self.prev_interoception = self.get_interoception()

        self.objects = []
        existing = set()
        while len(self.objects) < self.n_food:
            x = self.wrapped_env.np_random.randint(-self.activity_range / 2,
                                                   self.activity_range / 2) * 2
            y = self.wrapped_env.np_random.randint(-self.activity_range / 2,
                                                   self.activity_range / 2) * 2
            z = 0.05

            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = ObjectClass.FOOD
            self.objects.append((x, y, z, typ))
            existing.add((x, y))

        return self.get_current_obs()

    def generate_new_object(self, type_gen: ObjectClass):
        existing = set()
        for object in self.objects:
            existing.add((object[0], object[1]))

        while True:
            x = self.wrapped_env.np_random.randint(-self.activity_range / 2,
                                                   self.activity_range / 2) * 2
            y = self.wrapped_env.np_random.randint(-self.activity_range / 2,
                                                   self.activity_range / 2) * 2
            if type_gen is ObjectClass.FOOD:
                z = 0.05

            if (x, y) in existing:
                continue
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            return (x, y, z, type_gen)

    def step(self, action: np.ndarray):

        motor_action = action[:-1]
        evaporative_action = action[-1]  # final action is used for the homeostatic action

        self.prev_interoception = self.get_interoception()
        _, inner_rew, done, info = self.wrapped_env.step(motor_action)
        info['inner_rew'] = inner_rew
        if done:
            info['outer_rew'] = 0
            info["interoception"] = self.get_interoception()
            return self.get_current_obs(), self.dying_cost, done, info  # give a -10 rew if the robot dies

        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]

        #  Default Metabolic update
        self.internal_state[InteroClass.ENERGY] -= self.default_metabolic_update

        # Food-Eating
        new_objs = []
        self.num_food_eaten = 0
        for obj in self.objects:
            ox, oy, z, typ = obj
            # object within zone!
            if typ is ObjectClass.FOOD:
                if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                    self.internal_state[InteroClass.ENERGY] += self.success_consumptive_act
                    self.num_food_eaten += 1
                    new_objs.append(self.generate_new_object(type_gen=typ))
                else:
                    new_objs.append(obj)

        self.objects = new_objs

        # Update thermal step
        self.thermal_model.step(motor_action=motor_action,
                                evaporative_action=evaporative_action,
                                temp_air_c=self.temp_air,
                                temp_earth_c=self.temp_earth,
                                climate=self._climate,
                                dt=self.dt)

        self.internal_state[InteroClass.TEMPERATURE] = self.scale_temperature(self.thermal_model.get_temp_now())

        info["interoception"] = self.get_interoception()

        done = np.max(np.abs(self.get_interoception())) > self.survival_area

        self._step += 1
        done = done or self._step >= self._max_episode_steps

        reward, info_rew = self.get_reward(reward_setting=self.reward_setting,
                                 action=action,
                                 done=done)

        info.update(info_rew)

        return self.get_current_obs(), reward, done, info

    def get_reward(self, reward_setting, action, done):
        # Motor Cost
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = -.5 * np.square(action / scaling).sum()

        # Local Posture Cost
        if self.wrapped_env.IS_WALKER:
            euler = qtoeuler(self.wrapped_env.sim.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4])
            euler_stand = qtoeuler([1.0, 0.0, 0.0, 0.0])  # quaternion of standing state
            head_angle_cost = -np.square(euler[:2] - euler_stand[:2]).sum()  # Drop yaw
        else:
            head_angle_cost = 0.

        total_cost = self.coef_ctrl_cost * ctrl_cost + self.coef_head_angle * head_angle_cost

        # Main Reward
        info = {"reward_module": None}

        def drive(intero, target):
            drive_module = -1 * (intero - target) ** 2
            d_ = drive_module.sum()
            return d_, drive_module

        if reward_setting == "homeostatic":
            d, dm = drive(self.prev_interoception, self._target_internal_state)
            main_reward = d
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])

        elif reward_setting == "homeostatic_shaped":
            d, dm = drive(self.get_interoception(), self._target_internal_state)
            d_prev, dm_prev = drive(self.prev_interoception, self._target_internal_state)
            main_reward = d - d_prev
            info["reward_module"] = np.concatenate([self.coef_main_rew * (dm - dm_prev), [total_cost]])

        elif reward_setting == "one":
            # From continual-Cartpole setting from the lecture of Doina Precup (EEML 2021).
            if done:
                main_reward = -1.
            else:
                main_reward = 0.

        elif reward_setting == "homeostatic_biased":
            d, dm = drive(self.prev_interoception, self._target_internal_state)
            main_reward = d + self.reward_bias
            info["reward_module"] = np.concatenate([self.coef_main_rew * dm, [total_cost]])

        else:
            raise ValueError

        reward = self.coef_main_rew * main_reward + total_cost

        return reward, info

    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        food_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori()

        for ox, oy, oz, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5

            # only include readings for objects within range
            if dist > self.sensor_range:
                continue

            angle = math.atan2(oy - robot_y, ox - robot_x) - ori

            if math.isnan(angle):
                import ipdb;
                ipdb.set_trace()

            angle = angle % (2 * math.pi)

            if angle > math.pi:
                angle = angle - 2 * math.pi

            if angle < -math.pi:
                angle = angle + 2 * math.pi

            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5

            if abs(angle) > half_span:
                continue

            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range

            if typ is ObjectClass.FOOD:
                food_readings[bin_number] = intensity

        return food_readings

    def get_interoception(self):
        return np.array(list(self.internal_state.values()))

    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()

    def climate(self, climate_name=None):
        if climate_name == "sunny":
            return Climate.sunny
        elif climate_name == "cloudy":
            return Climate.cloudy
        elif climate_name == "default":
            return Climate.default
        else:
            raise ValueError("climate name must be any of sunny, cloudy, or default.")

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs()
        food_readings = self.get_readings()
        interoception = self.get_interoception()
        return np.concatenate([self_obs, food_readings, interoception])

    def get_climate(self):
        return copy(self._climate)

    @property
    def multi_modal_dims(self):
        self_obs_dim = len(self.wrapped_env.get_current_obs())

        food_readings = self.get_readings()
        readings_dim = len(food_readings)

        # (proprioception, exteroception, interoception)
        interoception_dim = len(self.get_interoception())
        return tuple([self_obs_dim, readings_dim, interoception_dim])

    @property
    def observation_space(self):
        return self.obs_space

    # space of only the robot observations (they go first in the get current obs)
    @property
    def robot_observation_space(self):
        return self.robot_obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def dt(self):
        return self.wrapped_env.dt

    def seed(self, seed=None):
        return self.wrapped_env.seed(seed)

    def scale_temperature(self, temperature):
        """
        Scale the temperature in Celsius degree into the range [-1, 1]
        :param temperature: absolute temperature
        :return:
        """
        out = 2 * (temperature - self.temp_limit_min) / (self.temp_limit_max - self.temp_limit_min) - 1
        return out

    def decode_temperature(self, scaled_temperature):
        """
        decode the scaled temperature [-1, 1] into temperature in Celsius degree
        :param scaled_temperature: temperature in scale of [-1, +1]
        :return:
        """
        out = 0.5 * (scaled_temperature + 1) * (self.temp_limit_max - self.temp_limit_min) + self.temp_limit_min
        return out

    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped env. If not successfull, falls
        back to the default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        obj = self.wrapped_env
        while not hasattr(obj, 'get_ori') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        try:
            return obj.get_ori()
        except (NotImplementedError, AttributeError) as e:
            pass
        return self.wrapped_env.sim.data.qpos[self.__class__.ORI_IND]

    def close(self):
        if self.wrapped_env.viewer:
            try:
                glfw.destroy_window(self.wrapped_env.viewer.window)
            except AttributeError:
                pass
            self.viewer = None

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):

        assert mode in {"human", "rgb_array"}, "depth is not supported."

        if self.wrapped_env.viewer is None:
            self.wrapped_env.render(mode, width, height, camera_id, camera_name)

        # Show Sensor Range
        if self.show_sensor_range:

            robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
            ori = self.get_ori()

            sensor_range = np.linspace(start=-self.sensor_span * 0.5,
                                       stop=self.sensor_span * 0.5,
                                       num=self.n_bins,
                                       endpoint=True)
            for direction in sensor_range:
                ox = robot_x + self.sensor_range * math.cos(direction + ori)
                oy = robot_y + self.sensor_range * math.sin(direction + ori)
                self.wrapped_env.viewer.add_marker(pos=np.array([ox, oy, 0.5]),
                                                   label=" ",
                                                   type=const.GEOM_SPHERE,
                                                   size=(0.1, 0.1, 0.1),
                                                   rgba=(0, 1, 0, 0.8))

        # Show Internal State
        if mode == "human":
            self.wrapped_env.viewer.add_overlay(
                const.GRID_TOPRIGHT, "ENERGY", f"{self.internal_state[InteroClass.ENERGY]:.4f}"
            )
            temp = self.decode_temperature(self.internal_state[InteroClass.TEMPERATURE])
            self.wrapped_env.viewer.add_overlay(
                const.GRID_TOPRIGHT, "CORE TEMP", f"{temp:.3f}"
            )

        # Show body core temperature
        if self.visualize_temp:
            raw_temp = np.clip(2 * self.get_interoception()[1], -1, 1)
            col = np.sign(self.get_interoception()[1])
            robot_x, robot_y, robo_z = self.wrapped_env.get_body_com("torso")[:3]
            self.wrapped_env.viewer.add_marker(pos=np.array([robot_x, robot_y, robo_z]),
                                               label=" ",
                                               type=const.GEOM_SPHERE,
                                               size=(.3, .3, .3),
                                               rgba=(col, 0, 1 - col, 0.8 * np.abs(raw_temp)))

        # Show food
        for obj in self.objects:
            ox, oy, oz, typ = obj
            if typ is ObjectClass.FOOD:
                self.wrapped_env.viewer.add_marker(pos=np.array([ox, oy, oz - 0.5]),
                                                   label=" ",
                                                   type=const.GEOM_SPHERE,
                                                   size=(0.5, 0.5, 0.5),
                                                   rgba=(1, 0, 0, 1))

        im = self.wrapped_env.render(mode, width, height, camera_id, camera_name)

        # delete unnecessary markers: https://github.com/openai/mujoco-py/issues/423#issuecomment-513846867
        del self.wrapped_env.viewer._markers[:]

        return im
