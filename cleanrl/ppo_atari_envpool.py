# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.envpool_stress import apply_stress_preset_to_args


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
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
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
    clip_coef: float = 0.1
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

    # Rsubt stability certificate arguments
    rsubt_monitor: bool = False
    """if toggled, enables Rsubt stability certificate monitoring"""
    rsubt_control: bool = False
    """if toggled, enables risk-gated hyperparameter control based on Rsubt"""
    rsubt_sketch_dim: int = 32768
    """dimension of the sketch space for Rsubt (larger for CNN)"""
    rsubt_buffer_size: int = 32
    """number of updates to store in Rsubt buffer"""
    rsubt_k_max: int = 10
    """maximum number of eigenvalues to track"""
    rsubt_diag_interval: int = 10
    """DEPRECATED: compute Rsubt diagnostics every N iterations (use diag_interval_steps instead)"""
    rsubt_diag_interval_steps: int = 20480
    """compute Rsubt diagnostics every N environment steps (step-based, consistent across num_envs)"""
    rsubt_min_buffer_entries: int = 4
    """minimum buffer entries before computing Rsubt (controls warmup speed)"""

    # Periodic evaluation arguments
    eval_interval_steps: int = 50000
    """evaluate every N environment steps (0 to disable)"""
    eval_episodes: int = 10
    """number of episodes per evaluation"""
    eval_seed: int = 12345
    """fixed seed for evaluation reproducibility"""
    eval_deterministic: bool = True
    """use deterministic (argmax) actions for evaluation"""

    # Stress test arguments
    stress_preset: str = ""
    """stress test preset name (see cleanrl_utils.stress_wrappers.STRESS_PRESETS)"""
    obs_noise_std: float = 0.0
    """observation noise std in normalized space (0..1); applied as noise*255 before model input"""
    action_delay: int = 0
    """action delay in steps (0 to disable)"""
    reward_scale: float = 1.0
    """reward scaling factor"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


class StressTransforms:
    """Array-level stress transforms for EnvPool Atari PPO."""

    def __init__(self, num_envs: int, obs_noise_std: float, action_delay: int, reward_scale: float, seed: int):
        self.num_envs = num_envs
        self.obs_noise_std = float(obs_noise_std)
        self.action_delay = int(action_delay)
        self.reward_scale = float(reward_scale)
        self.rng = np.random.default_rng(seed)
        if self.action_delay > 0:
            # queue of actions; each element is (num_envs,) int array
            self._action_queue = [None] * self.action_delay
        else:
            self._action_queue = None

    def apply_obs_noise_pixels(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian obs noise in normalized space [0,1] by adding (noise*255) to pixel tensor.
        This preserves the existing Agent behavior (it divides by 255 internally).
        """
        if self.obs_noise_std <= 0:
            return obs_tensor
        noise = self.rng.normal(0.0, self.obs_noise_std, size=obs_tensor.shape).astype(np.float32)
        noise_t = torch.from_numpy(noise).to(obs_tensor.device) * 255.0
        out = obs_tensor + noise_t
        return torch.clamp(out, 0.0, 255.0)

    def delay_action(self, action_np: np.ndarray) -> np.ndarray:
        if self._action_queue is None:
            return action_np
        if self._action_queue[0] is None:
            for i in range(len(self._action_queue)):
                self._action_queue[i] = np.zeros_like(action_np)
        delayed = self._action_queue[0]
        self._action_queue = self._action_queue[1:] + [action_np.copy()]
        return delayed

    def scale_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.reward_scale == 1.0:
            return reward
        return reward * self.reward_scale


def evaluate_atari_envpool(actor: nn.Module, env_id: str, episodes: int, seed: int, device: torch.device, deterministic: bool) -> tuple[float, float, float]:
    """Evaluate policy on EnvPool Atari with episodic_life=False and reward_clip=False."""
    env = envpool.make(env_id, env_type="gym", num_envs=1, seed=seed, episodic_life=False, reward_clip=False)
    returns = []
    lengths = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
            with torch.no_grad():
                logits = actor.get_action_and_value(obs_t)[0]  # action
                # If deterministic: argmax over logits; else sampling already happened.
                if deterministic:
                    # Recompute logits deterministically
                    hidden = actor.network(obs_t / 255.0)
                    raw_logits = actor.actor(hidden)
                    action = torch.argmax(raw_logits, dim=1)
                else:
                    action = logits
            obs, rew, done_arr, _info = env.step(action.cpu().numpy())
            done = bool(done_arr[0])
            ep_ret += float(rew[0])
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)
    env.close()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    apply_stress_preset_to_args(args)
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_text("original_env_id", args.env_id)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Stress transforms + episode tracking for true game-over returns
    stress = StressTransforms(
        num_envs=args.num_envs,
        obs_noise_std=args.obs_noise_std,
        action_delay=args.action_delay,
        reward_scale=args.reward_scale,
        seed=args.seed,
    )
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    # Rsubt stability certificate setup
    rsubt_monitor = None
    if args.rsubt_monitor:
        from cleanrl_utils.rsubt import RsubtMonitor
        actor_params = list(agent.network.parameters()) + list(agent.actor.parameters())
        rsubt_monitor = RsubtMonitor(
            params=actor_params,
            sketch_dim=args.rsubt_sketch_dim,
            buffer_size=args.rsubt_buffer_size,
            k_max=args.rsubt_k_max,
            diag_interval=args.rsubt_diag_interval,
            diag_interval_steps=args.rsubt_diag_interval_steps,
            min_buffer_entries=args.rsubt_min_buffer_entries,
            algorithm="ppo",
            device=device,
            seed=args.seed,
            enable_controller=args.rsubt_control,
            base_lr=args.learning_rate,
            base_epochs=args.update_epochs,
            base_clip=args.clip_coef,
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    last_eval_step = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # Apply obs noise stress in pixel space before storage and policy
            noisy_obs = stress.apply_obs_noise_pixels(next_obs)
            obs[step] = noisy_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(noisy_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            action_np = action.cpu().numpy()
            action_np = stress.delay_action(action_np)
            next_obs, reward, next_done, info = envs.step(action_np)
            reward = stress.scale_reward(np.asarray(reward, dtype=np.float32))
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Track true episode return on game over (terminated & lives==0)
            ep_returns += reward
            ep_lengths += 1
            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={ep_returns[idx]}")
                    avg_returns.append(ep_returns[idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", ep_returns[idx], global_step)
                    writer.add_scalar("charts/episodic_length", ep_lengths[idx], global_step)
                    ep_returns[idx] = 0.0
                    ep_lengths[idx] = 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # Rsubt: snapshot actor params before optimization
        if rsubt_monitor is not None:
            actor_params = list(agent.network.parameters()) + list(agent.actor.parameters())
            rsubt_monitor.snapshot_params(actor_params)

        # Rsubt: optionally apply risk-gated hyperparameters
        current_update_epochs = args.update_epochs
        current_clip_coef = args.clip_coef
        if rsubt_monitor is not None and args.rsubt_control:
            hp = rsubt_monitor.get_hyperparams()
            if hp is not None:
                current_update_epochs = hp.update_epochs
                current_clip_coef = hp.clip_coef
                optimizer.param_groups[0]["lr"] = hp.learning_rate
        for epoch in range(current_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > current_clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - current_clip_coef, 1 + current_clip_coef)
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
                # Baseline diagnostics: gradient norms
                actor_params_list = list(agent.network.parameters()) + list(agent.actor.parameters())
                critic_params_list = list(agent.critic.parameters())
                grad_norm_actor = sum(p.grad.norm().item() ** 2 for p in actor_params_list if p.grad is not None) ** 0.5
                grad_norm_critic = sum(p.grad.norm().item() ** 2 for p in critic_params_list if p.grad is not None) ** 0.5

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Rsubt: record update and compute certificate
        if rsubt_monitor is not None:
            actor_params = list(agent.network.parameters()) + list(agent.actor.parameters())
            rsubt_monitor.update(actor_params, global_step)
            rsubt_metrics = rsubt_monitor.maybe_compute(global_step)
            if rsubt_metrics is not None:
                rsubt_monitor.log_to_tensorboard(writer, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("baseline/grad_norm_actor", grad_norm_actor, global_step)
        writer.add_scalar("baseline/grad_norm_critic", grad_norm_critic, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Periodic evaluation
        if args.eval_interval_steps > 0 and (global_step - last_eval_step) >= args.eval_interval_steps:
            ret_mean, ret_std, len_mean = evaluate_atari_envpool(
                agent,
                args.env_id,
                args.eval_episodes,
                args.eval_seed,
                device,
                args.eval_deterministic,
            )
            writer.add_scalar("eval/return_mean", ret_mean, global_step)
            writer.add_scalar("eval/return_std", ret_std, global_step)
            writer.add_scalar("eval/length_mean", len_mean, global_step)
            print(f"eval: global_step={global_step}, return_mean={ret_mean:.2f}, return_std={ret_std:.2f}")
            last_eval_step = global_step

    envs.close()
    writer.close()
