# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import math
import os
import random
import time
from distutils.util import strtobool

import gym
import neurogym
import trp_env
import tiny_homeostasis
import thermal_regulation

import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from util import layer_init, BetaHead, make_env, test_env, create_connectivity_matrix, dual_test_env, check_percent_grad


def parse_args():
	# fmt: off
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
						help="the name of this experiment")
	parser.add_argument("--seed", type=int, default=1,
						help="seed of the experiment")
	parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
						help="if toggled, `torch.backends.cudnn.deterministic=False`")
	parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
						help="if toggled, cuda will be enabled by default")
	parser.add_argument("--gpu", type=int, default=0,
						help="GPU id if possible")
	parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
						help="if toggled, this experiment will be tracked with Weights and Biases")
	parser.add_argument("--wandb-project-name", type=str, default="ibs_ijcnn2023",
						help="the wandb's project name")
	parser.add_argument("--wandb-entity", type=str, default=None,
						help="the entity (team) of wandb's project")
	parser.add_argument("--wandb-group", type=str, default=None,
						help="the group of this run")
	parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
						help="weather to capture videos of the agent performances (check out `videos` folder)")

	# Algorithm specific arguments
	parser.add_argument("--env-id", type=str, default="SmallLowGearAntTRP-v0",
						help="the id of the environment")
	parser.add_argument("--test-every-itr", type=int, default=10,
						help="the agent is tested every this # of iterations")
	parser.add_argument("--n-test-runs", type=int, default=5,
						help="# of test runs (average)")
	parser.add_argument("--max-test-steps", type=int, default=60_000,
						help="maximum time steps in test runs")
	parser.add_argument("--num-test-envs", type=int, default=5,
						help="the number of parallel test game environments")
	parser.add_argument("--max-steps", type=int, default=60_000,
						help="maximum time steps in env runs")
	parser.add_argument("--total-timesteps", type=int, default=150_000_000,
						help="total timesteps of the experiments")
	parser.add_argument("--learning-rate", type=float, default=3e-4,
						help="the learning rate of the optimizer")
	parser.add_argument("--num-envs", type=int, default=1,
						help="the number of parallel game environments")
	parser.add_argument("--num-steps", type=int, default=300_000,
						help="the number of steps to run in each environment per policy rollout")
	parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
						help="Toggle learning rate annealing for policy and value networks")
	parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
						help="Use GAE for advantage computation")
	parser.add_argument("--gamma", type=float, default=0.99,
						help="the discount factor gamma")
	parser.add_argument("--gae-lambda", type=float, default=0.95,
						help="the lambda for the general advantage estimation")
	parser.add_argument("--num-minibatches", type=int, default=6,
						help="the number of mini-batches")
	parser.add_argument("--update-epochs", type=int, default=30,
						help="the K epochs to update the policy")
	parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
						help="Toggles advantages normalization")
	parser.add_argument("--clip-coef", type=float, default=0.3,
						help="the surrogate clipping coefficient")
	parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
						help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
	parser.add_argument("--ent-coef", type=float, default=0.001,
						help="coefficient of the entropy")
	parser.add_argument("--vf-coef", type=float, default=0.5,
						help="coefficient of the value function")
	parser.add_argument("--max-grad-norm", type=float, default=0.5,
						help="the maximum norm for the gradient clipping")
	parser.add_argument("--target-kl", type=float, default=None,
						help="the target KL divergence threshold")
 
	parser.add_argument("--gaussian-policy", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,)
	args = parser.parse_args()
	args.batch_size = int(args.num_envs * args.num_steps)
	args.minibatch_size = int(args.batch_size // args.num_minibatches)
	# fmt: on
	return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, gaussian=False):
        super().__init__()
        self.is_gaussian = gaussian
        
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            BetaHead(256, act_dim),
        )
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        probs = self.actor(x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def expand_layer(self, network, layer_idx, new_out_features, new_in_features):
        old_layer = network[layer_idx]
        in_features = old_layer.in_features
        old_out_features = old_layer.out_features

        if isinstance(old_layer, nn.Linear):
            if new_in_features == -1:
                new_in_features = in_features

            new_layer = nn.Linear(new_in_features, new_out_features)
            with torch.no_grad():
                new_layer.weight[:old_out_features, :in_features] = old_layer.weight.data
                new_layer.bias[:old_out_features] = old_layer.bias.data

            # Only the new part should be trainable
            def freeze_old_weights_hook(grad):
                grad[:old_out_features, :in_features] = 0
                return grad
            
            def freeze_old_bias_hook(grad):
                grad[:old_out_features] = 0
                return grad

            new_layer.weight.register_hook(freeze_old_weights_hook)
            new_layer.bias.register_hook(freeze_old_bias_hook)

            # Initialize new weights
            if new_in_features != in_features:
                nn.init.xavier_uniform_(new_layer.weight.data[old_out_features:, in_features:])
            else:
                nn.init.xavier_uniform_(new_layer.weight.data[old_out_features:, :])
            nn.init.zeros_(new_layer.bias.data[old_out_features:])

        elif isinstance(old_layer, BetaHead):
            new_layer = BetaHead(in_features, new_out_features)
            with torch.no_grad():
                new_layer.fcc_c0.weight[:old_out_features, :] = old_layer.fcc_c0.weight.data
                new_layer.fcc_c0.bias[:old_out_features] = old_layer.fcc_c0.bias.data
                new_layer.fcc_c1.weight[:old_out_features, :] = old_layer.fcc_c1.weight.data
                new_layer.fcc_c1.bias[:old_out_features] = old_layer.fcc_c1.bias.data

            def freeze_betahead_hook(grad):
                grad[:old_out_features] = 0
                return grad

            new_layer.fcc_c0.weight.register_hook(lambda g: g.index_fill(0, torch.arange(old_out_features), 0))
            new_layer.fcc_c0.bias.register_hook(freeze_betahead_hook)
            new_layer.fcc_c1.weight.register_hook(lambda g: g.index_fill(0, torch.arange(old_out_features), 0))
            new_layer.fcc_c1.bias.register_hook(freeze_betahead_hook)

            nn.init.xavier_uniform_(new_layer.fcc_c0.weight.data[old_out_features:, :])
            nn.init.zeros_(new_layer.fcc_c0.bias.data[old_out_features:])
            nn.init.xavier_uniform_(new_layer.fcc_c1.weight.data[old_out_features:, :])
            nn.init.zeros_(new_layer.fcc_c1.bias.data[old_out_features:])

        network[layer_idx] = new_layer

        # Update next layer if Linear
        next_layer_idx = layer_idx + 2
        if next_layer_idx < len(network):
            next_layer = network[next_layer_idx]
            if isinstance(next_layer, nn.Linear):
                old_next_weights = next_layer.weight.data.clone()
                old_next_bias = next_layer.bias.data.clone()

                new_next_layer = nn.Linear(new_out_features, next_layer.out_features)
                with torch.no_grad():
                    new_next_layer.weight[:, :old_out_features] = old_next_weights
                    new_next_layer.bias = nn.Parameter(old_next_bias)

                def freeze_old_inputs_hook(grad):
                    grad[:, :old_out_features] = 0
                    return grad

                new_next_layer.weight.register_hook(freeze_old_inputs_hook)

                nn.init.xavier_uniform_(new_next_layer.weight.data[:, old_out_features:])
                network[next_layer_idx] = new_next_layer
            elif isinstance(next_layer, BetaHead):
                next_layer.expand(new_in_features=new_out_features, new_out_features=next_layer.out_features)


    def expand_actor(self, layer_idx, new_out_features, new_in_features):
        self.expand_layer(self.actor, layer_idx, new_out_features, new_in_features)

    def expand_critic(self, layer_idx, new_out_features, new_in_features):
        self.expand_layer(self.critic, layer_idx, new_out_features, new_in_features)
        

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.cuda else "cpu")
    if not args.cuda:
        torch.set_num_threads(1)

    # env setup
    seed_ = (args.num_envs + args.num_test_envs) * args.seed
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=args.env_id,
                    seed=seed_ + i,
                    idx=i,
                    capture_video=args.capture_video,
                    run_name=run_name,
                    max_episode_steps=args.max_steps,
                    gaussian_policy=args.gaussian_policy) for i in range(args.num_envs)]
    )
    cog_envs = gym.vector.SyncVectorEnv(
        [make_env(env_id='PerceptualDecisionMaking-v0',
                    seed=seed_ + i,
                    idx=i,
                    capture_video=False,
                    run_name=None,
                    max_episode_steps=args.max_steps,
                    gaussian_policy=True,
                    cog=True) for i in range(args.num_envs)]
    )
    test_envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=args.env_id,
                    seed=seed_ + i + args.num_test_envs,
                    idx=i,
                    capture_video=False,
                    run_name=None,
                    max_episode_steps=args.max_test_steps,
                    gaussian_policy=args.gaussian_policy) for i in range(args.num_test_envs)]
    )
    
    cog_test_envs = gym.vector.SyncVectorEnv(
        [make_env(env_id='PerceptualDecisionMaking-v0',
                    seed=seed_ + i + args.num_test_envs,
                    idx=i,
                    capture_video=False,
                    run_name=None,
                    max_episode_steps=args.max_steps,
                    gaussian_policy=True,
                    cog=True) for i in range(args.num_test_envs)]
    )   

    agent = Agent(envs, gaussian=args.gaussian_policy).to(device)
    agent.load_state_dict(torch.load('./models/SmallLowGearAntTRP-v0__ppo__0__1741093440.pth', map_location=torch.device('cpu')))
    #freeze the parameters
    for param in agent.parameters():
        param.requires_grad = False
    
    agent.expand_actor(0,300, 69 + 3)
    agent.expand_actor(2,300, -1)
    agent.expand_actor(4,9, -1)
    agent.actor.insert(4, nn.Linear(300, 300))
    nn.init.xavier_uniform_(agent.actor[4].weight.data)
    nn.init.zeros_(agent.actor[4].bias.data)
    
    agent.expand_critic(0,300, 69 + 3)
    agent.expand_critic(2,300, -1)
    agent.expand_critic(4,1, -1)
    agent.critic.insert(4, nn.Linear(300, 300))
    nn.init.xavier_uniform_(agent.critic[4].weight.data)
    nn.init.zeros_(agent.critic[4].bias.data)
    
    agent = agent.to(device)
    
    percentage_grad = check_percent_grad(agent)
    print(f"Percentage of parameters with gradients: {percentage_grad:.2f}%")
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # agent = Agent(envs, gaussian=args.gaussian_policy).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_cog = (envs.single_observation_space.shape[0] + 3,)
    action_cog = (envs.single_action_space.shape[0] + 1,)   
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_cog).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_cog).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # ALGO Logic: Storage setup
    print("obs shape", obs.shape)
    print("actions shape", actions.shape)
    print(envs.single_observation_space.shape, envs.single_observation_space)
    print(envs.single_action_space.shape, envs.single_action_space)

	# TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_cog_obs = torch.Tensor(cog_envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_cog_done = torch.zeros(args.num_envs).to(device)
    
    next_obs = torch.cat((next_obs, next_cog_obs), dim=1)
    # take an or condition between next_done and next_cog_done
    next_done = torch.logical_or(next_done, next_cog_done)
    num_updates = args.total_timesteps // args.batch_size
    
 
	
    os.makedirs("models", exist_ok=True)

    for update in range(1, num_updates + 1):
        
        if update % (num_updates // 5) == 1:
            C = create_connectivity_matrix(agent.actor)
            #save the connectivity matrix
            # np.save(f"./connectivity_matrix/connectivity_matrix_{update}.npy", C)
        # Test Run every several iterations
        if update == 1 or update % args.test_every_itr == 0:
            print(f"Test @ {update - 1} START ------- ")
            episode_reward, episode_length, episode_error, ave_reward = dual_test_env(agent, test_envs, cog_test_envs,
                                                                                    n_runs=args.n_test_runs, device=device)
            print(
                f"########### TEST-{update - 1}: ave_episode_reward:{episode_reward}, ave_episode_length:{episode_length}, ave_episode_error:{episode_error}")
            writer.add_scalar("test/episodic_return", episode_reward, global_step)
            writer.add_scalar("test/episodic_length", episode_length, global_step)
            writer.add_scalar("test/episodic_intero_error", episode_error, global_step)
            writer.add_scalar("test/average_reward", ave_reward, global_step)
            writer.add_scalar("test/test_tick", update - 1, global_step)
            torch.save(agent.state_dict(), f"models/{run_name}_cog.pth")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy()[:, 8])
            next_cog_obs, cog_reward, cog_done, cog_info = cog_envs.step(action.cpu().numpy()[:,-1])
            
            reward = reward + 0.5 * cog_reward
            next_obs = torch.Tensor(torch.cat((torch.Tensor(next_obs), torch.Tensor(next_cog_obs)), dim=1)).to(device)
            next_done = torch.logical_or(torch.Tensor(done).to(device), torch.Tensor(cog_done).to(device))
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    nextvalues = nextvalues * nextnonterminal + values[t] * (1 - nextnonterminal)
                    delta = rewards[t] + args.gamma * nextvalues - values[t]  # homeostatic RL tips

                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        # print(envs.single_observation_space.shape)
        b_obs = obs.reshape((-1,) + (envs.single_observation_space.shape[0] + 3,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (envs.single_action_space.shape[0] + 1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
