# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import env
from rm import RewardMachine

@dataclass
class Args:
	exp_name: str = os.path.basename(__file__)[: -len(".py")]
	"""the name of this experiment"""
	seed: int = 1
	"""seed of the experiment"""
	torch_deterministic: bool = True
	"""if toggled, `torch.backends.cudnn.deterministic=False`"""
	cuda: bool = True
	"""if toggled, cuda will be enabled by default"""
	track: bool = False
	"""if toggled, this experiment will be tracked with Weights and Biases"""
	wandb_project_name: str = "cleanRL"
	"""the wandb's project name"""
	wandb_entity: str = None
	"""the entity (team) of wandb's project"""
	capture_video: bool = False
	"""whether to capture videos of the agent performances (check out `videos` folder)"""

	# Algorithm specific arguments
	env_id: str = "CartPole-v1"
	"""the id of the environment"""
	total_timesteps: int = 500000
	"""total timesteps of the experiments"""
	learning_rate: float = 2.5e-4
	"""the learning rate of the optimizer"""
	num_envs: int = 4
	"""the number of parallel game environments"""
	num_steps: int = 128
	"""the number of steps to run in each environment per policy rollout"""
	anneal_lr: bool = True
	"""Toggle learning rate annealing for policy and value networks"""
	gamma: float = 0.99
	"""the discount factor gamma"""
	gae_lambda: float = 0.95
	"""the lambda for the general advantage estimation"""
	num_minibatches: int = 4
	"""the number of mini-batches"""
	update_epochs: int = 4
	"""the K epochs to update the policy"""
	norm_adv: bool = True
	"""Toggles advantages normalization"""
	clip_coef: float = 0.2
	"""the surrogate clipping coefficient"""
	clip_vloss: bool = True
	"""Toggles whether or not to use a clipped loss for the value function, as per the paper."""
	ent_coef: float = 0.01
	"""coefficient of the entropy"""
	vf_coef: float = 0.5
	"""coefficient of the value function"""
	max_grad_norm: float = 0.5
	"""the maximum norm for the gradient clipping"""
	target_kl: float = None
	"""the target KL divergence threshold"""

	# to be filled in runtime
	batch_size: int = 0
	"""the batch size (computed in runtime)"""
	minibatch_size: int = 0
	"""the mini-batch size (computed in runtime)"""
	num_iterations: int = 0
	"""the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, reward_machine):
	def thunk():
		if capture_video and idx == 0:
			env = gym.make(env_id, render_mode="rgb_array", reward_machine=reward_machine)
			env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
		else:
			env = gym.make(env_id, reward_machine=reward_machine)
		env = gym.wrappers.RecordEpisodeStatistics(env)
		return env

	return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.orthogonal_(layer.weight, std)
	torch.nn.init.constant_(layer.bias, bias_const)
	return layer


class Agent(nn.Module):
	def __init__(self, envs):
		super().__init__()
		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, 1), std=1.0),
		)
		self.actor = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
		)

	def get_value(self, x):
		return self.critic(x)

	def get_action_and_value(self, x, action=None):
		logits = self.actor(x)
		probs = Categorical(logits=logits)
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
	args = tyro.cli(Args)
	args.batch_size = int(args.num_envs * args.num_steps)
	args.minibatch_size = int(args.batch_size // args.num_minibatches)
	args.num_iterations = args.total_timesteps // args.batch_size
	run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
	
	if args.track:
		import wandb

		wandb.init(
			project=args.wandb_project_name,
			entity=args.wandb_entity,
			sync_tensorboard=True,
			config=vars(args),
			name=run_name,
			monitor_gym=True,
			save_code=True,
		)

		# override args with sweep config
		args.learning_rate = wandb.config.get("learning_rate", args.learning_rate)
		args.num_steps = wandb.config.get("num_steps", args.num_steps)
		args.gamma = wandb.config.get("gamma", args.gamma)
		args.gae_lambda = wandb.config.get("gae_lambda", args.gae_lambda)
		args.ent_coef = wandb.config.get("ent_coef", args.ent_coef)
		args.clip_coef = wandb.config.get("clip_coef", args.clip_coef)
		args.update_epochs = wandb.config.get("update_epochs", args.update_epochs)
		args.num_minibatches = wandb.config.get("num_minibatches", args.num_minibatches)
		
		args.batch_size = int(args.num_envs * args.num_steps)
		args.minibatch_size = int(args.batch_size // args.num_minibatches)
		args.num_iterations = args.total_timesteps // args.batch_size
		 
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

	device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
 
	# RMs setup
	reward_machines = [RewardMachine() for _ in range(args.num_envs)]
 
	# env setup
	envs = gym.vector.SyncVectorEnv(
		[make_env(args.env_id, i, args.capture_video, run_name, reward_machines[i]) for i in range(args.num_envs)],
	)
	assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

	# agent = Agent(envs).to(device)
	# optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

	# TODO: check length
	rm_state_list = RewardMachine.get_states()
	rm_state_to_idx = {s: i for i, s in enumerate(rm_state_list)}
	idx_to_rm_state = {i: s for i, s in enumerate(rm_state_list)}
	print(reward_machines[0].get_current_state())
	print(rm_state_list)

	agent = {}
	for state in rm_state_list:
		agent[state] = Agent(envs).to(device)
	optimizer = {}
	for state in rm_state_list:
		optimizer[state] = optim.Adam(agent[state].parameters(), lr=args.learning_rate, eps=1e-5)
		
	# ALGO Logic: Storage setup
	obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
	actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
	logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
	rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
	dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
	values = torch.zeros((args.num_steps, args.num_envs)).to(device)
	rm_states = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
	
 	# TRY NOT TO MODIFY: start the game
	global_step = 0
	start_time = time.time()
	next_obs, _ = envs.reset(seed=args.seed)
	next_obs = torch.Tensor(next_obs).to(device)
	next_done = torch.zeros(args.num_envs).to(device)
 
	rm_state = reward_machines.get_current_state()

	for iteration in range(1, args.num_iterations + 1):
		# print("Iteration:", iteration)
		# Annealing the rate if instructed to do so.
		if args.anneal_lr:
			frac = 1.0 - (iteration - 1.0) / args.num_iterations
			lrnow = frac * args.learning_rate
			for state in envs.get_reward_machine().get_states():
				optimizer[state].param_groups[0]["lr"] = lrnow
				
		for step in range(0, args.num_steps):
			global_step += args.num_envs
			obs[step] = next_obs
			dones[step] = next_done
			rm_states[step] = rm_state_to_idx[rm_state]

			# ALGO LOGIC: action logic
			with torch.no_grad():
				action, logprob, _, value = agent[rm_state].get_action_and_value(next_obs)
				values[step] = value.flatten()
			actions[step] = action
			logprobs[step] = logprob

			# TRY NOT TO MODIFY: execute the game and log data.
			next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
			rm_state = envs.get_reward_machine().get_current_state()
			next_done = np.logical_or(terminations, truncations)
			rewards[step] = torch.tensor(reward).to(device).view(-1)
			next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

			if "_episode" in infos:
				for i, info in enumerate(infos["_episode"]):
					if info:
						# print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}, episodic_length={infos['episode']['l'][i]}")
						writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
						writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

		# bootstrap value if not done
		with torch.no_grad():
			next_value = agent[rm_state].get_value(next_obs).reshape(1, -1)
			advantages = torch.zeros_like(rewards).to(device)
			lastgaelam = 0
			for t in reversed(range(args.num_steps)):
				if t == args.num_steps - 1:
					nextnonterminal = 1.0 - next_done
					nextvalues = next_value
					rm_changed = (rm_states[t] != rm_state_to_idx[rm_state]).float()
				else:
					nextnonterminal = 1.0 - dones[t + 1]
					nextvalues = values[t + 1]
					rm_changed = (rm_states[t] != rm_states[t + 1]).float()
				
				# treat RM state change like an episode boundary for value bootstrap
				nextnonterminal = nextnonterminal * (1.0 - rm_changed)
				
				delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
				advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
			returns = advantages + values

		# flatten the batch
		b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
		b_logprobs = logprobs.reshape(-1)
		b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
		b_advantages = advantages.reshape(-1)
		b_returns = returns.reshape(-1)
		b_values = values.reshape(-1)
		b_rm_states = rm_states.reshape(-1)

		# Optimizing the policy and value network
		b_inds = np.arange(args.batch_size)
		# clipfracs = []
		all_clipfracs = {u:[] for u in rm_state_list}
		all_pg_loss = {u:None for u in rm_state_list}
		all_v_loss = {u:None for u in rm_state_list}
		all_entropy_loss = {u:None for u in rm_state_list}
		all_old_approx_kl = {u:None for u in rm_state_list}
		all_approx_kl = {u:None for u in rm_state_list}
		
		for epoch in range(args.update_epochs):
			np.random.shuffle(b_inds)
			for start in range(0, args.batch_size, args.minibatch_size):
				end = start + args.minibatch_size
				mb_inds = b_inds[start:end]
				
				# Update each agent only on transitions from its RM state
				for u in rm_state_list:
					
					# Filter for transitions where the RM state is u
					rm_mask = (b_rm_states[mb_inds] == rm_state_to_idx[u])
					if rm_mask.sum() == 0:
						continue
					
					u_inds = mb_inds[rm_mask]

					_, newlogprob, entropy, newvalue = agent[u].get_action_and_value(b_obs[u_inds], b_actions.long()[u_inds])
					logratio = newlogprob - b_logprobs[u_inds]
					ratio = logratio.exp()

					with torch.no_grad():
						# calculate approx_kl http://joschu.net/blog/kl-approx.html
						old_approx_kl = (-logratio).mean()
						approx_kl = ((ratio - 1) - logratio).mean()
						all_clipfracs[u] += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
						all_old_approx_kl[u] = old_approx_kl
						all_approx_kl[u] = approx_kl

					mb_advantages = b_advantages[u_inds]
					if args.norm_adv:
						if mb_advantages.numel() > 1 and mb_advantages.std() > 1e-8:
							mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

					# Policy loss
					pg_loss1 = -mb_advantages * ratio
					pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
					pg_loss = torch.max(pg_loss1, pg_loss2).mean()

					# Value loss
					newvalue = newvalue.view(-1)
					if args.clip_vloss:
						v_loss_unclipped = (newvalue - b_returns[u_inds]) ** 2
						v_clipped = b_values[u_inds] + torch.clamp(
							newvalue - b_values[u_inds],
							-args.clip_coef,
							args.clip_coef,
						)
						v_loss_clipped = (v_clipped - b_returns[u_inds]) ** 2
						v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
						v_loss = 0.5 * v_loss_max.mean()
					else:
						v_loss = 0.5 * ((newvalue - b_returns[u_inds]) ** 2).mean()

					entropy_loss = entropy.mean()
					loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

					optimizer[u].zero_grad()
					loss.backward()
					nn.utils.clip_grad_norm_(agent[u].parameters(), args.max_grad_norm)
					optimizer[u].step()
					
					all_pg_loss[u] = pg_loss
					all_v_loss[u] = v_loss
					all_entropy_loss[u] = entropy_loss

			if args.target_kl is not None:
				if any(all_approx_kl[u] is not None and all_approx_kl[u] > args.target_kl for u in rm_state_list):
					break

		y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
		var_y = np.var(y_true)
		explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

		# TRY NOT TO MODIFY: record rewards for plotting purposes
		# Logging - log metrics per RM state
		for u in rm_state_list:
			writer.add_scalar(f"charts/learning_rate/agent_{u}", optimizer[u].param_groups[0]["lr"], global_step)
			if all_v_loss[u] is not None:
				writer.add_scalar(f"losses/value_loss/agent_{u}", all_v_loss[u].item(), global_step)
				writer.add_scalar(f"losses/policy_loss/agent_{u}", all_pg_loss[u].item(), global_step)
				writer.add_scalar(f"losses/entropy/agent_{u}", all_entropy_loss[u].item(), global_step)
				writer.add_scalar(f"losses/approx_kl/agent_{u}", all_approx_kl[u].item(), global_step)
				writer.add_scalar(f"losses/clipfrac/agent_{u}", np.mean(all_clipfracs[u]), global_step)
				
		writer.add_scalar("losses/explained_variance", explained_var, global_step)
		print("SPS:", int(global_step / (time.time() - start_time)))
		writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

	envs.close()
	writer.close()
