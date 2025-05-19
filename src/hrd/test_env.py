import gym
import trp_env
import tiny_homeostasis
import thermal_regulation
# import trp_cog_env

import numpy as np
import torch
import torch.nn as nn
import math
from hrl_bs_ijcnn2023.util import layer_init, BetaHead, make_env

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Agent(nn.Module):
	def __init__(self, envs, gaussian=False):
		super().__init__()

		self.is_gaussian = gaussian
		self.envs = envs

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

	def get_position(self):
		# Get the position of the agent in the environment
		return self.envs.wrapped_env.get_body_com("torso")[:2]

	def get_value(self, x):
		return self.critic(x)

	def get_action_and_value(self, x, action=None):
		probs = self.actor(x)
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action), probs.entropy(), self.critic(x)

	def get_activations(self, x):
		activations_actor = []
		activations_critic = []
		x_actor = x
		x_critic = x
		for layer in self.actor:
			if isinstance(layer, nn.Linear):
				x_actor = layer(x_actor)
				activations_actor.append(x_actor)

		for layer in self.critic:
			if isinstance(layer, nn.Linear):
				x_critic = layer(x_critic)
				activations_critic.append(x_critic)
    
		return activations_actor, activations_critic

def plot_umap(activations, title):
	import umap
	import matplotlib.pyplot as plt

	reducer = umap.UMAP()
 
	embedding = reducer.fit_transform(activations[0])
	embedding_random = reducer.fit_transform(activations[1])

	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	ax[0].scatter(embedding[:, 0], embedding[:, 1], s=1)
	ax[0].set_title("UMAP of first layer activations")
	ax[0].set_xlabel("UMAP1")
	ax[0].set_ylabel("UMAP2")
	ax[1].scatter(embedding_random[:, 0], embedding_random[:, 1], s=1)
	ax[1].set_title("UMAP of random first layer activations")
	ax[1].set_xlabel("UMAP1")
	ax[1].set_ylabel("UMAP2")
	plt.show()
 

nut_values = [-0.7, -0.3, 0.0, 0.3, 0.7]
dir = "./hrl_bs_ijcnn2023/"
n_activations = []
n_activations_random = []
n_positions = []
n_velocities = []

for n in range(len(nut_values)):
	seed_ = (1 + 1) * 1
	envs = gym.vector.SyncVectorEnv(
		[make_env(env_id='SmallLowGearAntTRP-v0',
					seed=seed_ + i,
					idx=i,
					capture_video=False,
					run_name='test',
					max_episode_steps=60_000,
					gaussian_policy=False,
					nutrient_val=[0.5,0.5]) for i in range(1)]
	)


	agent = Agent(envs=envs, gaussian=False)


	agent.load_state_dict(torch.load("./hrl_bs_ijcnn2023/models/SmallLowGearAntTRP-v0__ppo__0__1741201287.pth", weights_only=True))
	agent_random = Agent(envs=envs, gaussian=False)

	env = gym.make("SmallLowGearAntTRP-v0", internal_reset = "setpoint", nutrient_val=[nut_values[n], nut_values[n]],)
	env.seed(100)  # Seeding

	done = False

	obs = env.reset()
	max_episode_steps = 3000
	# action = env.action_space.sample()
	actions = []
	obss = []
	step = 0


	all_activations = []
	all_activations_random = []
	positions = []

	while step < max_episode_steps:
		obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

		position = env.wrapped_env.get_body_com("torso")[:2]
		print(f"Step: {step}, Position: {position}")
  
		action = agent.get_action_and_value(obs)[0].detach().numpy()[0]
		action_random = agent_random.get_action_and_value(obs)[0].detach().numpy()[0]
		activations_actor, activations_critic = agent.get_activations(obs)
		activations_random = agent_random.get_activations(obs)
		layer1_activation_actor = activations_actor[0].detach().numpy()[0]
		layer1_activation_critic = activations_critic[0].detach().numpy()[0]
		
		all_activations.append(layer1_activation_actor)
		all_activations_random.append(layer1_activation_critic)
	
		#scale the action to the action space between env.action_space.low and env.action_space.high
		action  = action * (env.action_space.high - env.action_space.low) + env.action_space.low
		
		actions.append(action)
		obss.append(obs.detach().numpy())
		positions.append(position.copy())
		
		obs, reward, done, info = env.step(action)

		env.render()
		step += 1
		
	env.close()

	# Convert to numpy arrays
	activations_np = np.array(all_activations)          # shape: [timesteps, features]
	activations_random_np = np.array(all_activations_random)
	positions = np.array(positions)  # shape: [timesteps, 2]
	velocities = np.array([positions[i] - positions[i-1] for i in range(1, len(positions), 1)])
	speeds = np.array([np.sqrt(np.square(velocities[j][0]) + np.square(velocities[j][1])) for j in range(len(velocities))]) # Calculate speed as the norm of the velocity vector
 
	n_activations.append(activations_np)
	n_activations_random.append(activations_random_np)
	
	n_positions.append(positions)
	n_velocities.append(speeds)
 
n_activations = np.array(n_activations)  
n_activations_random = np.array(n_activations_random) 
n_positions = np.array(positions) 
 
#normalize the activations between -1 and 1
n_activations = (n_activations - np.min(n_activations)) / (np.max(n_activations) - np.min(n_activations)) * 2 - 1
n_activations_random = (n_activations_random - np.min(n_activations_random)) / (np.max(n_activations_random) - np.min(n_activations_random)) * 2 - 1

pca_result = n_activations
pca_result_random = n_activations_random

fig, axs = plt.subplots(5, 3, figsize=(18, 21), sharey=True)

for i in range(len(nut_values)):
    # Plot the heatmap for the trained agent
	axs[i, 0].imshow(pca_result[i].T, aspect='auto', cmap='viridis', interpolation='nearest')
	axs[i, 0].set_title(f"Actor's Second Layer Activation (Nutrient: {nut_values[i]})")
	axs[i, 0].set_xlabel("Time Step")
	axs[i, 0].set_ylabel("Neuron Index")	
	axs[i, 0].set_yticks(range(pca_result.shape[2]))  # Show all neuron indices
    # Plot the heatmap for the random agent
	axs[i, 1].imshow(pca_result_random[i].T, aspect='auto', cmap='viridis', interpolation='nearest')
	axs[i, 1].set_title(f"Critic's Second Layer Activation (Nutrient: {nut_values[i]})")
	axs[i, 1].set_xlabel("Time Step")
	axs[i, 1].set_ylabel("Neuron Index")
	axs[i, 1].set_yticks(range(pca_result_random.shape[2]))  # Show all neuron indices
 
	#plot the velocities
	axs[i, 2].plot(n_velocities[i], c='blue', alpha=0.5, label='Velocity')
	axs[i, 2].set_title(f"Velocity (Nutrient: {nut_values[i]})")
	axs[i, 2].set_xlabel("Time Step")
	axs[i, 2].set_ylabel("Velocity")
	axs[i, 2].set_ylim(0, np.max(n_velocities) * 1.1)  # Set y-axis limits to show velocity range
	axs[i, 2].grid(True)

 
plt.tight_layout()	
plt.savefig("neural_activations_layer1.png", dpi=300)
