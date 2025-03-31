import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from util import layer_init, BetaHead, make_env

import gym
import trp_env
import tiny_homeostasis
import thermal_regulation

def create_connectivity_matrix(layers: nn.Sequential):
    
	total_neurons = 0
	for layer in layers:
		if isinstance(layer, nn.Linear):
			# layer = layers[i]
			total_neurons += layer.out_features
		
	total_neurons += layers[0].in_features
	print(total_neurons)
    # total_neurons =
	C = np.zeros((total_neurons, total_neurons))

	start_idx = 0  # Index of first neuron in the current layer
	for layer in layers:
		# layer = layers[i]
		if isinstance(layer, nn.Linear):
			C[start_idx:start_idx + layer.in_features, start_idx + layer.in_features:start_idx + layer.in_features + layer.out_features] = layer.weight.T.detach().numpy()
			C.T[start_idx:start_idx + layer.in_features, start_idx + layer.in_features:start_idx + layer.in_features + layer.out_features] = layer.weight.T.detach().numpy()
			start_idx += layer.in_features
   
	return C

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

# agent = Agent(envs=envs,
#               mixture_vf=False,
#               attention_vf=False,
#               mixture_policy=False,
#               attention_policy=False)

agent = Agent(envs=envs, gaussian=False)

print("## plot before loading model weights")
C1 = create_connectivity_matrix(agent.actor)

#load the model
agent.load_state_dict(torch.load('./models/SmallLowGearAntTRP-v0__ppo__0__1741093440.pth'))
print(agent.actor)
# print(agent.actor[-1].fcc_c0.weight)
layer1 = agent.actor[0]
# plot 10 random neurons
fig, axs = plt.subplots(1, 10, figsize=(20, 5))
for i in range(10):
	axs[i].plot(layer1.weight[i].detach().numpy())
plt.show()

plt.clf()
plt.close()
# Example usage
# layer_sizes = [3, 5, 4, 2]  # 3 input, 5 hidden, 4 hidden, 2 output
C2 = create_connectivity_matrix(agent.actor)
# # print(C)

#plot before and after side by side with same scale
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(C1, cmap='coolwarm', interpolation='nearest')
axs[0].set_title('Before')
axs[1].imshow(C2, cmap='coolwarm', interpolation='nearest')
axs[1].set_title('After')
plt.tight_layout()
# plt.savefig('connectivity_before_vs_after.png')
plt.show()
