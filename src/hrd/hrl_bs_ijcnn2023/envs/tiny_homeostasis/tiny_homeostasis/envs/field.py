from collections import deque

import numpy as np

import gym
from gym.spaces import Box


class FieldEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self,
				 dim_resource=2,
				 max_episode_steps=np.inf,
				 internal_reset="random",
				 internal_random_range=(-1. / 6, 1. / 6),
				 *args, **kwargs):

		self.dim_resource = dim_resource
		self._max_episode_steps = max_episode_steps
		self.internal_reset = internal_reset
		self.internal_random_range = internal_random_range

		self.action_space = Box(low=-1, high=1, shape=(2,))
		self.observation_space = Box(low=-1, high=1, shape=(4+2*dim_resource,))

		# env const
		self.reward_scale = 100.
		self.dt = 1
		self.resource_decay = 0.001
		self.decay_inlet_vel = 0.5
		self.scale_pos_accel = 0.5
		self.resource_inlet = 0.02
		self.pos_limit = 1.
		self.vel_limit = 2.
		self.vel_decay = 0.3
		self.vel_inlet_limit = 0.1

		self.area_range = 0.2
		if self.dim_resource < 2:
			self.resource_pos_list = [np.array([1, 0])]
		self.resource_pos_list = [np.array([0.4 * np.cos(i * 2 * np.pi / self.dim_resource),
											0.4 * np.sin(i * 2 * np.pi / self.dim_resource)]) for i in range(self.dim_resource)]

		# env variables
		self.resource = np.zeros(dim_resource)
		self.resource_prev = self.resource.copy()
		self.vel_resource = np.zeros(dim_resource)
		self.vel_inlet = np.zeros(dim_resource)
		self.pos = np.zeros(2)
		self.vel = np.zeros(2)
		self._step = 0

		# visualization
		self.global_step = 0
		self.fig = None
		self.n_hist = 100
		self.pos_hist = deque(maxlen=self.n_hist)
		self.intero_hist = deque(maxlen=self.n_hist)
		self.intero_error_hist = deque(maxlen=self.n_hist)

	@property
	def dim_intero(self):
		return np.prod(self.resource.shape)

	def reset(self):
		self._step = 0

		if self.internal_reset == "random":
			self.resource = np.random.uniform(self.internal_random_range[0], self.internal_random_range[1], self.dim_resource)
		else:
			self.resource = np.zeros(self.dim_resource)

		self.resource_prev = self.resource.copy()
		self.vel_resource = np.zeros(self.dim_resource)
		self.pos = np.random.uniform(-1, 1, 2)
		self.vel = np.zeros(2)
		return self.get_obs()

	def get_obs(self):
		return np.concatenate([
			self.pos / self.pos_limit,
			self.vel / self.vel_limit,
			self.vel_resource,
			self.resource,  # last dims are interoception!
		], axis=0)

	def get_inlet(self):
		inlets = np.zeros(self.dim_resource)

		for i, pos in enumerate(self.resource_pos_list):
			if np.linalg.norm(self.pos - pos, 2) < self.area_range:
				inlets[i] = 1

		return self.resource_inlet * inlets, inlets.astype(np.bool8)

	def get_reward(self):
		# Keramati & Gutkin 2014

		def drive(res):
			drive_module = res ** 2
			drive_sum = drive_module.sum()
			return drive_sum, drive_module

		d, dm = drive(self.resource)
		d_prev, dm_prev = drive(self.resource_prev)

		return self.reward_scale * (d_prev - d), self.reward_scale * (dm_prev - dm), dm

	def step(self, action: float):
		# position update
		self.vel = self.vel_decay * self.vel + self.scale_pos_accel * action * self.dt
		self.vel = np.clip(self.vel, a_min=-self.vel_limit, a_max=self.vel_limit)

		self.pos += self.vel * self.dt
		self.pos = np.clip(self.pos, a_min=-self.pos_limit, a_max=self.pos_limit)

		inlet, inlet_flag = self.get_inlet()

		self.vel_inlet = self.decay_inlet_vel * self.vel_inlet + inlet
		self.vel_inlet = np.clip(self.vel_inlet, a_min=-self.vel_inlet_limit, a_max=self.vel_inlet_limit)
		self.vel_resource = self.vel_inlet - self.resource_decay

		self.resource_prev = self.resource.copy()
		self.resource += self.vel_resource * self.dt

		reward, reward_module, intero_error = self.get_reward()

		done = False
		if np.any(np.abs(self.resource) > 1.0):
			done = True

		self._step += 1
		done = done or self._step >= self._max_episode_steps

		info = {
			"interoception": self.resource,
			"reward_module": reward_module,
			"vel_resource": self.vel_resource,
			"inlet_flag": inlet_flag,
			"intero_error": intero_error.sum(),
			"pos": self.pos,
			"vel": self.vel
		}

		self.pos_hist.append(info["pos"].copy())
		self.intero_hist.append(info["interoception"].copy())
		self.intero_error_hist.append(np.abs(info["interoception"].copy()))

		self.global_step += 1

		return self.get_obs(), reward, done, info

	def plot_field(self, pyplot):
		angle = np.arange(0, 2 * np.pi, 0.1)
		x_ = self.area_range * np.cos(angle)
		y_ = self.area_range * np.sin(angle)
		for i, pos in enumerate(self.resource_pos_list):
			pyplot.scatter(pos[0], pos[1], s=20)
			pyplot.plot(pos[0] + x_, pos[1] + y_, c="k")

		ph = np.array(self.pos_hist)

		pyplot.scatter(ph[-10:, 0], ph[-10:, 1], s=5, c="r", alpha=0.3)
		pyplot.plot(ph[-10:, 0], ph[-10:, 1], c="r", alpha=0.3)

		pyplot.xlim([-1.1, 1.1])
		pyplot.ylim([-1.1, 1.1])

	def plot_resource(self, pyplot):
		ih = np.array(self.intero_hist)

		for i in range(self.dim_resource):
			pyplot.plot(ih[:, i])
		pyplot.ylim([-1.1, +1.1])

	def plot_error(self, pyplot):
		ieh = np.array(self.intero_error_hist)

		for i in range(self.dim_resource):
			pyplot.plot(ieh[:, i])
		pyplot.ylim([0, 0.5])

	def render(self, mode="human"):
		import matplotlib.pyplot as plt

		if self.fig is None:
			self.fig = plt.figure()

		plt.subplot(311)
		plt.cla()
		self.plot_field(plt)

		plt.subplot(312)
		plt.cla()
		self.plot_resource(plt)

		plt.subplot(313)
		plt.cla()
		self.plot_error(plt)

		plt.pause(0.001)
