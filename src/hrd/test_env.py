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

env = gym.make("SmallLowGearAntTRP-v0", internal_reset = "setpoint", nutrient_val=[0.7,0.7])

env.seed(100)  # Seeding

done = False

obs = env.reset()
print(env.action_space.high, env.action_space.low)
print(env.multi_modal_dims)
max_episode_steps = 2000
# action = env.action_space.sample()
actions = []
obss = []
step = 0


all_activations = []
all_activations_random = []

while not done and step < max_episode_steps:
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    # print(obs.shape)
    # print(step)
    
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
    
    obs, reward, done, info = env.step(action)

    env.render()
    step += 1
    
env.close()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Convert to numpy arrays
activations_np = np.array(all_activations)          # shape: [timesteps, features]
activations_random_np = np.array(all_activations_random)

print(activations_np.shape)
# Run PCA
pca = PCA(n_components=30)
pca_result = pca.fit_transform(activations_np)
pca_result_random = pca.fit_transform(activations_random_np)

print(pca_result.shape)

# Plotting PCA colored by time
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# # plt.scatter(pca_result[:, 0], pca_result[:, 1], c=np.arange(len(pca_result)), cmap='viridis', s=5)
# # plt.colorbar(label="Timestep")
# plt.plot(activations_np[:,:], c='blue', alpha=0.5, label='Trained Agent')
# plt.title("Trained Agent Activations Over Time")
# plt.xlabel("time")
# plt.ylabel("Neural Activity")

# After converting to numpy arrays (right before your PCA code):
# Convert to numpy arrays
# activations_np = np.array(all_activations)          # shape: [timesteps, features]
# activations_random_np = np.array(all_activations_random)

# Plot heatmaps
plt.figure(figsize=(15, 6))

# Trained Agent Heatmap
plt.subplot(1, 2, 1)
plt.imshow(pca_result.T, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Activation Strength')
plt.title("Actor's Second Layer Activations when full")
plt.xlabel("Time Step")
plt.ylabel("Neuron Index")
plt.yticks(range(pca_result.shape[1]))  # Show all neuron indices

# Random Agent Heatmap
plt.subplot(1, 2, 2)
plt.imshow(pca_result_random.T, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Activation Strength')
plt.title("Critic's Second Layer Activations when full")
plt.xlabel("Time Step")
plt.ylabel("Neuron Index")
plt.yticks(range(pca_result_random.shape[1]))

plt.tight_layout()
plt.savefig("pca_neural_activations_layer1_full.png", dpi=300)
# plt.show()


# plt.subplot(1, 2, 2)
# # plt.scatter(pca_result_random[:, 0], pca_result_random[:, 1], c=np.arange(len(pca_result_random)), cmap='viridis', s=5)
# # plt.colorbar(label="Timestep")
# plt.plot(pca_result[:, 1], c='red', alpha=0.5, label='Random Agent')
# plt.title("PCA of Random Agent Activations Over Time")
# plt.xlabel("time")
# plt.ylabel("PC2")

# plt.tight_layout()
# plt.show()

# plot_umap([all_activations, all_activations_random], "UMAP of first layer activations")

#plot and visualize relationship between obss[:, 27:] and first_layer[:, 27:] weights of the actor network
# use any visualization technique to show the relationship between the observation space and the first layer of the actor network
# obss = np.array(obss)
# obss = obss.squeeze(1)
# obss = obss[:256, 27:]

# layer_weight = agent.actor[0].weight.detach().numpy()
# layer_random = torch.nn.init.orthogonal_(torch.empty(256, 40), np.sqrt(2)).numpy()
# layer_weight = layer_weight[:, 27:]
# # print(obss[256, 27:].shape, layer_weight.shape)

# pc1, pc2 = PCA(n_components=2).fit_transform(obss[:, 27:]).T
# pc1_w, pc2_w = PCA(n_components=2).fit_transform(layer_weight).T
# pc_w_r, pc2_w_r = PCA(n_components=2).fit_transform(layer_random).T



# fig, ax = plt.subplots(1, 3, figsize=(10, 5))
# ax[0].scatter(pc1, pc2)
# ax[0].set_title("PCA of observation space")
# ax[0].set_xlabel("PC1")
# ax[0].set_ylabel("PC2")

# ax[1].scatter(pc1_w, pc2_w)
# ax[1].set_title("PCA of first layer weights")
# ax[1].set_xlabel("PC1")
# ax[1].set_ylabel("PC2")

# ax[2].scatter(pc_w_r, pc2_w_r)
# ax[2].set_title("PCA of random first layer weights")
# ax[2].set_xlabel("PC1")
# ax[2].set_ylabel("PC2")


# plt.show()

# #plot the eigen vectors as well  
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# eigen_values, eigen_vectors = np.linalg.eig(np.cov(layer_random.T))
# print(eigen_values)
# print(eigen_vectors)
# # eigen_values = np.sqrt(eigen_values)
# eigen_vectors = eigen_vectors.T

# eigen_values_w, eigen_vectors_w = np.linalg.eig(np.cov(layer_weight.T))
# eigen_values_w = np.sqrt(eigen_values_w)
# eigen_vectors_w = eigen_vectors_w.T

# ax[0].plot([0, eigen_vectors[0, 0]], [0, eigen_vectors[0, 1]], 'r')
# ax[0].plot([0, eigen_vectors[1, 0]], [0, eigen_vectors[1, 1]], 'b')
# ax[0].set_title("Eigen vectors of observation space")
# ax[0].set_xlabel("PC1")
# ax[0].set_ylabel("PC2")

# ax[1].plot([0, eigen_vectors_w[0, 0]], [0, eigen_vectors_w[0, 1]], 'r')
# ax[1].plot([0, eigen_vectors_w[1, 0]], [0, eigen_vectors_w[1, 1]], 'b')
# ax[1].set_title("Eigen vectors of first layer weights")
# ax[1].set_xlabel("PC1")
# ax[1].set_ylabel("PC2")

# plt.show()


# #plot the 2 principle components of the action space using PCA

# actions = np.array(actions)

# random_actions = np.random.uniform(-1, 1, (100, 8))

# pc1, pc2, pc3 = PCA(n_components=3).fit_transform(actions).T

# #plot random 100 points
# indices = np.random.choice(len(pc1), 100)
# pc1 = pc1[indices]
# pc2 = pc2[indices]
# pc3 = pc3[indices]

# #plot random and trained actions side by side
# ax, fig = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})


# pc1_r, pc2_r, pc3_r = PCA(n_components=3).fit_transform(random_actions).T
# fig[0].scatter(pc1_r, pc2_r, pc3_r)
# fig[0].set_title("PCA of random actions")
# # fig[0].set_xlabel("PC1")
# # fig[0].set_ylabel("PC2")
# # fig[0].set_zlabel("PC3")

# fig[1].scatter(pc1, pc2, pc3)
# fig[1].set_title("PCA of trained actions")
# # fig[1].set_xlabel("PC1")
# # fig[1].set_ylabel("PC2")
# # fig[1].set_zlabel("PC3")


# plt.show()
    

