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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse

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
			if isinstance(layer, (nn.Linear,BetaHead)):
				x_actor = layer(x_actor)
				if isinstance(layer, BetaHead):
					x_actor = x_actor.sample()
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
 
 
def main(max_steps=3000, velocity_per_step=10, plot_type='all'):
	print(f'Running test environment with max_steps={max_steps}')
	if plot_type == 'all':
		nut_values = [-0.7, -0.3, 0.0, 0.3, 0.7]
	elif plot_type == 'vel':
		nut_values = [-0.5]
	dir = "./hrl_bs_ijcnn2023/"
	n_activations = []
	n_activations_random = []
	n_activations_layer2_act = []
	n_activations_layer2_crt = []
	n_activations_output_act = []
	n_activations_output_crt = []
	n_positions = []
	n_velocities = []
	n_internal_mags = []
	max_episode_steps = max_steps
 
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
		
		# action = env.action_space.sample()
		actions = []
		obss = []
		step = 0


		all_activations = []
		all_activations_random = []
		all_activations_layer2_act = []
		all_activations_layer2_crt = []
		all_activations_output_act = []
		all_activations_output_crt = []
		positions = []
		internal_mags = []

		while step < max_episode_steps:
			obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

			position = env.wrapped_env.get_body_com("torso")[:2]
			print(f"Step: {step}, Position: {position}")
	
			action = agent.get_action_and_value(obs)[0].detach().numpy()[0]
			action_random = agent_random.get_action_and_value(obs)[0].detach().numpy()[0]
			activations_actor, activations_critic = agent.get_activations(obs)
					
			layer1_activation_actor = activations_actor[0].detach().numpy()[0]
			layer1_activation_critic = activations_critic[0].detach().numpy()[0]
			layer2_activation_actor = activations_actor[1].detach().numpy()[0]
			layer2_activation_critic = activations_critic[1].detach().numpy()[0]
			output_activation_actor = activations_actor[2].detach().numpy()[0]
			output_activation_critic = activations_critic[2].detach().numpy()[0]

			
			all_activations.append(layer1_activation_actor)
			all_activations_random.append(layer1_activation_critic)
			all_activations_layer2_act.append(layer2_activation_actor)
			all_activations_layer2_crt.append(layer2_activation_critic)
			all_activations_output_act.append(output_activation_actor)
			all_activations_output_crt.append(output_activation_critic)
		
			#scale the action to the action space between env.action_space.low and env.action_space.high
			action  = action * (env.action_space.high - env.action_space.low) + env.action_space.low
			
			actions.append(action)
			obss.append(obs.detach().numpy())
			positions.append(position.copy())
			
			obs, reward, done, info = env.step(action)
   
			internal_state = list(env.internal_state.values())
   
			internal_mag = np.sqrt(np.square(internal_state[0]) + np.square(internal_state[1]))
			internal_mags.append(internal_mag)
			
			env.render()
			step += 1
			
		env.close()

		# Convert to numpy arrays
		activations_np = np.array(all_activations)          # shape: [timesteps, features]
		activations_random_np = np.array(all_activations_random)
		activations_layer2_act_np = np.array(all_activations_layer2_act)  # shape: [timesteps, features]
		activations_layer2_crt_np = np.array(all_activations_layer2_crt)  # shape: [timesteps, features]
		activations_output_act_np = np.array(all_activations_output_act)  # shape: [timesteps, features]
		activations_output_crt_np = np.array(all_activations_output_crt)  # shape: [timesteps, features]
		positions = np.array(positions)  # shape: [timesteps, 2]
		velocities = np.array([positions[i-1] - positions[i-velocity_per_step] for i in range(velocity_per_step, len(positions), velocity_per_step)])
		speeds = np.array([np.sqrt(np.square(velocities[j][0]) + np.square(velocities[j][1])) for j in range(len(velocities))]) # Calculate speed as the norm of the velocity vector

		#add the initial velocity to the velocities as 0
		speeds = np.insert(speeds, 0, 0)
		n_activations.append(activations_np)
		n_activations_random.append(activations_random_np)
		n_activations_layer2_act.append(activations_layer2_act_np)
		n_activations_layer2_crt.append(activations_layer2_crt_np)
		n_activations_output_act.append(activations_output_act_np)
		n_activations_output_crt.append(activations_output_crt_np)
		n_internal_mags.append(internal_mags)
		
		n_positions.append(positions)
		n_velocities.append(speeds)
	
	n_activations = np.array(n_activations)  
	n_activations_random = np.array(n_activations_random)
	n_activations_layer2_act = np.array(n_activations_layer2_act)
	n_activations_layer2_crt = np.array(n_activations_layer2_crt)
	n_activations_output_act = np.array(n_activations_output_act)
	n_activations_output_crt = np.array(n_activations_output_crt)
	n_positions = np.array(positions) 
	n_velocities = np.array(n_velocities)
	#normalize the activations between -1 and 1
	n_activations = (n_activations - np.min(n_activations)) / (np.max(n_activations) - np.min(n_activations)) * 2 - 1
	n_activations_random = (n_activations_random - np.min(n_activations_random)) / (np.max(n_activations_random) - np.min(n_activations_random)) * 2 - 1

	pca_result = n_activations
	pca_result_random = n_activations_random

	#make velocity x axis as long as the number of timesteps in the activations
	t_steps = np.arange(0, max_steps - 1, velocity_per_step)
	t_steps_new = np.arange(max_steps)
	print(f't_steps shape: {t_steps.shape}, t_steps_new shape: {t_steps_new.shape}, n_velocities shape: {n_velocities.shape}')
	print(t_steps, t_steps_new)
	
 
	n_velocities_interp = []

	for v in range(len(n_velocities)):
		interpolator = interp1d(t_steps, n_velocities[v], kind="cubic", fill_value="extrapolate")
		velocity = interpolator(t_steps_new)
		n_velocities_interp.append(velocity)
		n_bins = 20
		bins = np.linspace(0, np.sqrt(2), n_bins + 1)
		bin_indices = np.digitize(internal_mags[v], bins)

		bin_centers = []
		velocity_vars = []
		for i in range(1, len(bins)):
			idxs = np.where(bin_indices == i)[0]
			if len(idxs) > 1:
				var = np.var(velocities[idxs])
				center = (bins[i] + bins[i-1]) / 2
				velocity_vars.append(var)
				bin_centers.append(center)

      
  
	n_velocities_interp = np.array(n_velocities_interp)
 
	if plot_type == 'all':
	
		# Swapped rows and columns
		fig, axs = plt.subplots(7, 5, figsize=(21, 28))  
		print(f'pca_result shape: {pca_result.shape}, pca_result_random shape: {pca_result_random.shape}, n_velocities_interp shape: {n_velocities_interp.shape}')

		for i in range(len(nut_values)):
			# Plot the heatmap for the trained agent layer 1 actor
			
			axs[0, i].imshow(pca_result[i].T, aspect='auto', cmap='viridis', interpolation='nearest')
			axs[0, i].set_title(f"Actor's Second Layer Activation\nNutrient: {nut_values[i]}")
			axs[0, i].set_xlabel("Time Step")
			axs[0, i].set_ylabel("Neuron Index")
			axs[0, i].set_yticks(np.arange(0, 256, 8))  # Show all neuron indices
			axs[0, i].set_yticklabels(np.arange(0, 256, 8)) 


			# Plot the heatmap for the trained agent layer 1 critic
			axs[1, i].imshow(pca_result_random[i].T, aspect='auto', cmap='viridis', interpolation='nearest')
			axs[1, i].set_title(f"Critic's Second Layer Activation\nNutrient: {nut_values[i]}")
			axs[1, i].set_xlabel("Time Step")
			axs[1, i].set_ylabel("Neuron Index")
			axs[1, i].set_yticks(np.arange(0, 256, 8))  # Show all neuron indices
			axs[1, i].set_yticklabels(np.arange(0, 256, 8)) 
   
			# Plot the heatmap for the layer 2 actor activations
			axs[2, i].imshow(n_activations_layer2_act[i].T, aspect='auto', cmap='viridis', interpolation='nearest')
			axs[2, i].set_title(f"Actor's Layer 2 Activation\nNutrient: {nut_values[i]}")
			axs[2, i].set_xlabel("Time Step")
			axs[2, i].set_ylabel("Neuron Index")
			axs[2, i].set_yticks(np.arange(0, 256, 8))
			axs[2, i].set_yticklabels(np.arange(0, 256, 8))
   
			# Plot the heatmap for the layer 2 critic activations
			axs[3, i].imshow(n_activations_layer2_crt[i].T, aspect='auto', cmap='viridis', interpolation='nearest')
			axs[3, i].set_title(f"Critic's Layer 2 Activation\nNutrient: {nut_values[i]}")
			axs[3, i].set_xlabel("Time Step")
			axs[3, i].set_ylabel("Neuron Index")
			axs[3, i].set_yticks(np.arange(0, 256, 8))
			axs[3, i].set_yticklabels(np.arange(0, 256, 8))
			# Plot the output layer actor activations
			axs[4, i].imshow(n_activations_output_act[i].T, aspect='auto', cmap='viridis', interpolation='nearest')
			axs[4, i].set_title(f"Actor's Output Layer Activation\nNutrient: {nut_values[i]}")
			axs[4, i].set_xlabel("Time Step")
			axs[4, i].set_ylabel("Neuron Index")	
			axs[4, i].set_yticks(np.arange(0, 8, 1))
			axs[4, i].set_yticklabels(np.arange(0, 8, 1))
			# Plot the output layer critic activations
			axs[5, i].plot(n_activations_output_crt[i], c='red', alpha=0.5, label='Critic Output Layer Activation')
			axs[5, i].set_title(f"Critic's Output Layer Activation\nNutrient: {nut_values[i]}")
			axs[5, i].set_xlabel("Time Step")
			axs[5, i].set_ylabel("Neuron Index")
			axs[5, i].set_xlim(0, max_steps - 1)
			axs[5, i].grid(True)

			# Plot the velocities
			axs[6, i].plot(n_velocities_interp[i], c='blue', alpha=0.5, label='Velocity')
			axs[6, i].set_title(f"Velocity\nNutrient: {nut_values[i]}")
			axs[6, i].set_xlabel("Time Step")
			axs[6, i].set_xlim(0, max_steps - 1)
			axs[6, i].set_ylabel("Velocity")
			axs[6, i].set_ylim(0, 0.8)
			axs[6, i].grid(True)

		import time
		plt.tight_layout()	
		plt.savefig("hrl_bs_ijcnn2023/plots/neural_activity/neural_activations_layer1" + str(time.time()) + ".png", dpi=300)
		plt.close()
  
		#save all the data as .npz file
		np.savez("hrl_bs_ijcnn2023/plots/neural_activity/neural_activations.npz",
			activations_layer1_act=n_activations,
			activations_layer1_crt=n_activations_random,
			activations_layer2_act=n_activations_layer2_act,
			activations_layer2_crt=n_activations_layer2_crt,
			activations_output_act=n_activations_output_act,
			activations_output_crt=n_activations_output_crt,
			positions=n_positions,
			velocities=n_velocities_interp,
			internal_mags=n_internal_mags
		)
	
	if plot_type == 'vel':
		print(bin_centers, velocity_vars)
		plt.plot(bin_centers, velocity_vars, marker='o')
		plt.xlabel("Internal Magnitude (sqrt(red² + blue²))")
		plt.ylabel("Velocity Variance")
		plt.title("Velocity Variance vs Internal State Magnitude")
		plt.grid(True)
		plt.savefig("hrl_bs_ijcnn2023/plots/neural_activity/velocity_variance_vs_internal_magnitude.png", dpi=300)
	
	
if __name__ == "__main__":
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Test environment for HRD")
	parser.add_argument('--max_steps', type=int, default=3000, help='Maximum number of steps per episode')
	parser.add_argument('--velocity_per_step', type=int, default=10, help='Number of steps to calculate velocity')
	parser.add_argument('--plot_type', type=str, default='all', choices=['all', 'vel'], help='Type of plot to generate')
	args = parser.parse_args()
	main(max_steps=args.max_steps, velocity_per_step=args.velocity_per_step, plot_type=args.plot_type)
