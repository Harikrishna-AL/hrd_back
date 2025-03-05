import gym
import trp_env
import tiny_homeostasis
import thermal_regulation

import numpy as np
import torch
import torch.nn as nn
import math
from hrl_bs_ijcnn2023.util import layer_init, BetaHead, make_env

class Agent(nn.Module):
	def __init__(self, envs, gaussian=False):
		super().__init__()

		self.is_gaussian = gaussian

		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
			nn.Tanh(),
			layer_init(nn.Linear(256, 256)),
			nn.Tanh(),
			layer_init(nn.Linear(256, 1), std=1.0),
		)

		self.actor = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
			nn.Tanh(),
			layer_init(nn.Linear(256, 256)),
			nn.Tanh(),
			BetaHead(256, np.prod(envs.single_action_space.shape)),
		)

	def get_value(self, x):
		return self.critic(x)

	def get_action_and_value(self, x, action=None):
		probs = self.actor(x)
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action), probs.entropy(), self.critic(x)


seed_ = (1 + 1) * 1
envs = gym.vector.SyncVectorEnv(
    [make_env(env_id='SmallLowGearAntTRP-v0',
                seed=seed_ + i,
                idx=i,
                capture_video=False,
                run_name='test',
                max_episode_steps=60_000,
                gaussian_policy=False) for i in range(1)]
)

agent = Agent(envs=envs, gaussian=False)
agent.load_state_dict(torch.load("./hrl_bs_ijcnn2023/models/SmallLowGearAntTRP-v0__ppo__0__1741093440.pth", weights_only=True))


env = gym.make("SmallLowGearAntTRP-v0")

env.seed(100)  # Seeding

done = False

obs = env.reset()
print(env.action_space.high, env.action_space.low)
print(env.multi_modal_dims)
# action = env.action_space.sample()

while not done:
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    # print(obs.shape)
    
    action = agent.get_action_and_value(obs)[0].detach().numpy()[0]
    #scale the action to the action space between env.action_space.low and env.action_space.high
    action  = action * (env.action_space.high - env.action_space.low) + env.action_space.low


    
    obs, reward, done, info = env.step(action)
    # print("#reward:",reward)
    
    env.render()
