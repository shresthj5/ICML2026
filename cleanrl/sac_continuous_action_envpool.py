# EnvPool-native SAC for continuous action spaces (MuJoCo + DMC)
# Compatible with cleanrl's TensorBoard logging contract for scored early-warning suite
import os
import random
import time
from dataclasses import dataclass

import envpool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.envpool_registry import get_envpool_id, get_envpool_backend, detect_env_type
from cleanrl_utils.envpool_vecenv import (
    flatten_dmc_observation,
    split_terminations_truncations,
    action_bounds_from_envpool_dm,
    action_bounds_from_envpool_gym,
    action_scale_bias,
)
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
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task (legacy or EnvPool-native)"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Rsubt stability certificate arguments
    rsubt_monitor: bool = False
    """if toggled, enables Rsubt stability certificate monitoring"""
    rsubt_control: bool = False
    """if toggled, enables risk-gated hyperparameter control based on Rsubt"""
    rsubt_sketch_dim: int = 8192
    """dimension of the sketch space for Rsubt"""
    rsubt_buffer_size: int = 64
    """number of updates to store in Rsubt buffer"""
    rsubt_k_max: int = 20
    """maximum number of eigenvalues to track"""
    rsubt_diag_interval: int = 10
    """DEPRECATED: compute Rsubt diagnostics every N actor updates (use diag_interval_steps instead)"""
    rsubt_diag_interval_steps: int = 20480
    """compute Rsubt diagnostics every N environment steps (step-based, consistent across num_envs)"""
    rsubt_min_buffer_entries: int = 4
    """minimum buffer entries before computing Rsubt (controls warmup speed)"""

    # Periodic evaluation arguments
    eval_interval_steps: int = 10000
    """evaluate every N environment steps (0 to disable)"""
    eval_episodes: int = 10
    """number of episodes per evaluation"""
    eval_seed: int = 12345
    """fixed seed for evaluation reproducibility"""
    eval_deterministic: bool = True
    """use deterministic actions for evaluation"""

    # Stress test arguments (array-level transforms for EnvPool)
    stress_preset: str = ""
    """stress test preset name (see cleanrl_utils.stress_wrappers.STRESS_PRESETS)"""
    obs_noise_std: float = 0.0
    """observation noise std (0 to disable)"""
    obs_dropout_p: float = 0.0
    """observation dropout probability (0 to disable)"""
    action_delay: int = 0
    """action delay in steps (0 to disable)"""
    reward_scale: float = 1.0
    """reward scaling factor"""


class RecordEpisodeStatistics:
    """Track episode statistics for EnvPool environments."""
    
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
    
    def reset(self):
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
    
    def update(self, rewards: np.ndarray, dones: np.ndarray):
        """Update statistics and return info for completed episodes."""
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        infos = []
        for i, done in enumerate(dones):
            if done:
                infos.append({
                    "episode": {
                        "r": self.episode_returns[i],
                        "l": self.episode_lengths[i],
                    }
                })
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0
            else:
                infos.append(None)
        return infos


class StressTransforms:
    """Array-level stress transforms for EnvPool environments."""
    
    def __init__(
        self,
        num_envs: int,
        obs_noise_std: float = 0.0,
        obs_dropout_p: float = 0.0,
        action_delay: int = 0,
        reward_scale: float = 1.0,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.obs_noise_std = obs_noise_std
        self.obs_dropout_p = obs_dropout_p
        self.action_delay = action_delay
        self.reward_scale = reward_scale
        # Running stats for relative noise (per-dimension)
        self._obs_mean = None
        self._obs_var = None
        self._obs_count = 0
        
        if action_delay > 0:
            self.action_queue = [None] * action_delay
        else:
            self.action_queue = None
    
    def transform_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_noise_std > 0:
            # Match ObsNoiseWrapper default: relative noise w.r.t. running std
            if self._obs_mean is None:
                self._obs_mean = np.zeros(obs.shape[1:], dtype=np.float32)
                self._obs_var = np.ones(obs.shape[1:], dtype=np.float32)
            batch = obs.astype(np.float32)
            self._obs_count += 1
            delta = batch.mean(axis=0) - self._obs_mean
            self._obs_mean += delta / max(self._obs_count, 1)
            self._obs_var += delta * (batch.mean(axis=0) - self._obs_mean) - self._obs_var / max(self._obs_count, 1)
            obs_std = np.sqrt(np.maximum(self._obs_var, 1e-8))
            noise_std = self.obs_noise_std * obs_std
            noise = self.rng.normal(0, noise_std, size=obs.shape).astype(obs.dtype)
            obs = obs + noise
        if self.obs_dropout_p > 0:
            mask = self.rng.random(obs.shape) > self.obs_dropout_p
            obs = obs * mask.astype(obs.dtype)
        return obs
    
    def transform_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.reward_scale != 1.0:
            reward = reward * self.reward_scale
        return reward
    
    def delay_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_queue is None:
            return action
        if self.action_queue[0] is None:
            for i in range(len(self.action_queue)):
                self.action_queue[i] = np.zeros_like(action)
        delayed_action = self.action_queue[0]
        self.action_queue = self.action_queue[1:] + [action.copy()]
        return delayed_action


class ReplayBuffer:
    """Simple replay buffer for off-policy learning."""
    
    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, obs, next_obs, action, reward, done):
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.observations[self.ptr] = obs[i]
            self.next_observations[self.ptr] = next_obs[i]
            self.actions[self.ptr] = action[i]
            self.rewards[self.ptr] = reward[i]
            self.dones[self.ptr] = done[i]
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.observations[idxs]).to(self.device),
            torch.FloatTensor(self.next_observations[idxs]).to(self.device),
            torch.FloatTensor(self.actions[idxs]).to(self.device),
            torch.FloatTensor(self.rewards[idxs]).to(self.device),
            torch.FloatTensor(self.dones[idxs]).to(self.device),
        )


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_scale: np.ndarray, action_bias: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def get_hidden(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return h2


def evaluate_agent(
    actor: Actor,
    envpool_id: str,
    backend: str,
    num_episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool = True,
) -> dict:
    """Run periodic evaluation using EnvPool environment (same as training).
    
    Uses EnvPool with num_envs=1 to ensure consistent observation shapes with training.
    """
    is_dmc = (backend == "dm")
    
    if is_dmc:
        eval_env = envpool.make_dm(envpool_id, num_envs=1, seed=seed)
        action_spec = eval_env.action_spec()
        action_low = float(action_spec.minimum.flat[0]) if hasattr(action_spec.minimum, 'flat') else float(action_spec.minimum)
        action_high = float(action_spec.maximum.flat[0]) if hasattr(action_spec.maximum, 'flat') else float(action_spec.maximum)
    else:
        eval_env = envpool.make(envpool_id, env_type="gym", num_envs=1, seed=seed)
        action_low = eval_env.action_space.low
        action_high = eval_env.action_space.high
    
    returns = []
    lengths = []
    
    for ep in range(num_episodes):
        # Reset environment
        if is_dmc:
            timestep = eval_env.reset()
            obs = flatten_dmc_observation(timestep.observation, 1).astype(np.float32)
            done = timestep.last()[0]
        else:
            obs = eval_env.reset()
            obs = obs.reshape(1, -1).astype(np.float32)
            done = False
        
        ep_return = 0.0
        ep_length = 0
        
        while not done:
            obs_tensor = torch.Tensor(obs[0]).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, mean_action = actor.get_action(obs_tensor)
                if deterministic:
                    action = mean_action
            action_np = action.cpu().numpy().flatten()
            action_np = np.clip(action_np, action_low, action_high)
            action_np = action_np.reshape(1, -1)  # EnvPool expects (num_envs, action_dim)
            
            # Step environment
            if is_dmc:
                timestep = eval_env.step(action_np)
                obs = flatten_dmc_observation(timestep.observation, 1).astype(np.float32)
                reward = timestep.reward[0]
                done = timestep.last()[0]
            else:
                obs, reward, done_arr, info = eval_env.step(action_np)
                obs = obs.reshape(1, -1).astype(np.float32)
                reward = reward[0]
                done = done_arr[0]
            
            ep_return += float(reward)
            ep_length += 1
            
            # Safety limit
            if ep_length >= 10000:
                break
        
        returns.append(ep_return)
        lengths.append(ep_length)
    
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    apply_stress_preset_to_args(args)
    
    # Convert env_id to EnvPool format
    envpool_id = get_envpool_id(args.env_id)
    backend = get_envpool_backend(envpool_id)
    env_type = detect_env_type(envpool_id)
    
    run_name = f"{envpool_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
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

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create EnvPool environment
    if backend == "dm":
        envs = envpool.make_dm(envpool_id, num_envs=args.num_envs, seed=args.seed)
        is_dmc = True
    else:
        envs = envpool.make(envpool_id, env_type="gym", num_envs=args.num_envs, seed=args.seed)
        is_dmc = False
    
    # Get observation and action dimensions
    if is_dmc:
        action_spec = envs.action_spec()
        action_low, action_high = action_bounds_from_envpool_dm(action_spec)
        action_dim = int(np.prod(action_spec.shape))
        ts0 = envs.reset()
        obs0 = flatten_dmc_observation(ts0.observation, args.num_envs)
        obs_dim = obs0.shape[1]
        timestep = envs.reset()
    else:
        action_low, action_high = action_bounds_from_envpool_gym(envs.action_space)
        action_dim = int(np.prod(envs.action_space.shape))
        obs_dim = int(np.prod(envs.observation_space.shape))
    
    action_scale, action_bias = action_scale_bias(action_low, action_high)

    # Initialize networks
    actor = Actor(obs_dim, action_dim, action_scale, action_bias).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Episode statistics and stress transforms
    episode_stats = RecordEpisodeStatistics(args.num_envs)
    stress_transforms = StressTransforms(
        num_envs=args.num_envs,
        obs_noise_std=args.obs_noise_std,
        obs_dropout_p=args.obs_dropout_p,
        action_delay=args.action_delay,
        reward_scale=args.reward_scale,
        seed=args.seed,
    )

    # Replay buffer
    rb = ReplayBuffer(args.buffer_size, obs_dim, action_dim, device)

    # Rsubt stability certificate setup
    rsubt_monitor = None
    if args.rsubt_monitor:
        from cleanrl_utils.rsubt import RsubtMonitor
        rsubt_monitor = RsubtMonitor(
            params=list(actor.parameters()),
            sketch_dim=args.rsubt_sketch_dim,
            buffer_size=args.rsubt_buffer_size,
            k_max=args.rsubt_k_max,
            diag_interval=args.rsubt_diag_interval,
            diag_interval_steps=args.rsubt_diag_interval_steps,
            min_buffer_entries=args.rsubt_min_buffer_entries,
            algorithm="sac",
            device=device,
            seed=args.seed,
            enable_controller=args.rsubt_control,
            base_lr=args.policy_lr,
            base_policy_freq=args.policy_frequency,
        )

    start_time = time.time()
    last_eval_step = 0
    grad_norm_actor = 0.0

    # Reset environment
    if is_dmc:
        obs = flatten_dmc_observation(timestep.observation, args.num_envs)
    else:
        obs = envs.reset()
        obs = obs.reshape(args.num_envs, -1)
    
    obs = stress_transforms.transform_obs(obs)
    episode_stats.reset()

    for global_step in range(args.total_timesteps):
        # RSUBT: optionally apply risk-gated hyperparameters (affects actor LR / policy frequency / skipping)
        current_policy_frequency = args.policy_frequency
        skip_actor = False
        if rsubt_monitor is not None and args.rsubt_control:
            hp = rsubt_monitor.get_hyperparams()
            if hp is not None:
                current_policy_frequency = hp.policy_frequency
                actor_optimizer.param_groups[0]["lr"] = hp.actor_lr
                skip_actor = hp.skip_actor

        # Action logic
        if global_step < args.learning_starts:
            actions = np.random.uniform(action_low, action_high, size=(args.num_envs, action_dim)).astype(np.float32)
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        
        actions = stress_transforms.delay_action(actions)
        actions = np.clip(actions, action_low, action_high)
        
        # Execute action
        if is_dmc:
            timestep = envs.step(actions)
            next_obs = flatten_dmc_observation(timestep.observation, args.num_envs)
            rewards = timestep.reward.astype(np.float32)
            done_log = timestep.last().astype(bool)
            dones = timestep.last().astype(np.float32)  # terminal for learning
        else:
            next_obs, rewards, dones, infos = envs.step(actions)
            next_obs = next_obs.reshape(args.num_envs, -1)
            # Match CleanRL: treat time-limit truncation as non-terminal for bootstrapping
            done_log = np.asarray(dones).astype(bool)
            split = split_terminations_truncations(dones, infos)
            dones = split.terminations.astype(np.float32)  # only true terminals
        
        next_obs = stress_transforms.transform_obs(next_obs)
        rewards = stress_transforms.transform_reward(rewards)
        
        # Track episode statistics
        ep_infos = episode_stats.update(rewards, done_log)
        for info in ep_infos:
            if info is not None:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        # Store transition
        rb.add(obs, next_obs, actions, rewards.reshape(-1, 1), dones.reshape(-1, 1))
        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(b_next_obs)
                qf1_next_target = qf1_target(b_next_obs, next_state_actions)
                qf2_next_target = qf2_target(b_next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = b_rewards.flatten() + (1 - b_dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(b_obs, b_actions).view(-1)
            qf2_a_values = qf2(b_obs, b_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % current_policy_frequency == 0:
                # Rsubt: snapshot actor params before update
                if rsubt_monitor is not None:
                    rsubt_monitor.snapshot_params(list(actor.parameters()))

                if not skip_actor:
                    for _ in range(current_policy_frequency):
                        pi, log_pi, _ = actor.get_action(b_obs)
                        qf1_pi = qf1(b_obs, pi)
                        qf2_pi = qf2(b_obs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        grad_norm_actor = (
                            sum(p.grad.norm().item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
                        )
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(b_obs)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # Rsubt: record update and compute certificate
                if rsubt_monitor is not None:
                    rsubt_monitor.update(list(actor.parameters()), global_step)
                    rsubt_metrics = rsubt_monitor.maybe_compute(global_step)
                    if rsubt_metrics is not None:
                        rsubt_monitor.log_to_tensorboard(writer, global_step)

            # Update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Logging
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("baseline/grad_norm_actor", grad_norm_actor, global_step)
                
                # Representation rank diagnostics
                if args.rsubt_monitor and global_step % 1000 == 0:
                    from cleanrl_utils.rsubt.reprank import compute_effective_rank, compute_stable_rank
                    with torch.no_grad():
                        actor_hidden = actor.get_hidden(b_obs)
                        actor_effrank = compute_effective_rank(actor_hidden)
                        actor_stablerank = compute_stable_rank(actor_hidden)
                    writer.add_scalar("reprank/actor_effrank", actor_effrank, global_step)
                    writer.add_scalar("reprank/actor_stablerank", actor_stablerank, global_step)
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Periodic evaluation
        if args.eval_interval_steps > 0 and (global_step - last_eval_step) >= args.eval_interval_steps:
            eval_stats = evaluate_agent(actor, envpool_id, backend, args.eval_episodes, args.eval_seed, device, args.eval_deterministic)
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)
            writer.add_scalar("eval/return_std", eval_stats["return_std"], global_step)
            writer.add_scalar("eval/length_mean", eval_stats["length_mean"], global_step)
            print(f"eval: global_step={global_step}, return_mean={eval_stats['return_mean']:.2f}")
            last_eval_step = global_step

    envs.close()
    writer.close()
