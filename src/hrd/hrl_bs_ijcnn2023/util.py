import gym

import numpy as np
import torch
from torch import nn


def make_env(env_id, seed, idx, capture_video, run_name, max_episode_steps=np.inf, gaussian_policy=False):
    def thunk():
        env = gym.make(
            env_id,
            dim_resource=3,
            max_episode_steps=max_episode_steps,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        if not gaussian_policy:
            env = gym.wrappers.RescaleAction(env, 0, 1)  # for Beta policy
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class BetaHead(nn.Module):
    def __init__(self, in_features, action_size):
        super(BetaHead, self).__init__()

        self.fcc_c0 = nn.Linear(in_features, action_size)
        nn.init.orthogonal_(self.fcc_c0.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c0.bias)

        self.fcc_c1 = nn.Linear(in_features, action_size)
        nn.init.orthogonal_(self.fcc_c1.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c1.bias)

    def forward(self, x):
        c0 = torch.nn.functional.softplus(self.fcc_c0(x)) + 1.
        c1 = torch.nn.functional.softplus(self.fcc_c1(x)) + 1.
        return torch.distributions.Independent(
            torch.distributions.Beta(c1, c0), 1
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer.bias, "data"):
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def test_env(agent: torch.nn.Module, test_envs: gym.vector.SyncVectorEnv, n_runs, device):
    agent.eval()

    count_episode = 0
    episode_reward = 0
    episode_length = 0
    episode_error = 0
    ave_reward = 0

    intero_errors = np.zeros(len(test_envs.envs))
    obs = torch.Tensor(test_envs.reset()).to(device)
    while count_episode < n_runs:

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)

        obs, reward, done, info = test_envs.step(action.cpu().numpy())
        obs = torch.Tensor(obs).to(device)

        for idx, item in enumerate(info):
            if "interoception" in item.keys():
                intero_errors[idx] += (item["interoception"] ** 2).sum()

            if "episode" in item.keys():
                print(
                    f"TEST: episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}, episodic_error={intero_errors[idx] / item['episode']['l']}")
                count_episode += 1
                episode_reward += item['episode']['r']
                episode_length += item['episode']['l']
                episode_error += intero_errors[idx] / item['episode']['l']
                ave_reward += item['episode']['r'] / item['episode']['l']
                intero_errors[idx] = 0

            if n_runs <= count_episode:
                break

    episode_reward /= n_runs
    episode_length /= n_runs
    episode_error /= n_runs
    ave_reward /= n_runs

    agent.train()

    return episode_reward, episode_length, episode_error, ave_reward
