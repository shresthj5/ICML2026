# EnvPool-native PPO for continuous action spaces (MuJoCo + DMC)
# Compatible with cleanrl's TensorBoard logging contract for scored early-warning suite
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import envpool
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.envpool_registry import get_envpool_id, get_envpool_backend, detect_env_type
from cleanrl_utils.envpool_vecenv import (
    flatten_dmc_observation,
    action_bounds_from_envpool_dm,
    action_bounds_from_envpool_gym,
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment (legacy or EnvPool-native)"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
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
    rsubt_sketch_dim: int = 8192
    """dimension of the sketch space for Rsubt"""
    rsubt_buffer_size: int = 64
    """number of updates to store in Rsubt buffer"""
    rsubt_k_max: int = 20
    """maximum number of eigenvalues to track"""
    rsubt_diag_interval: int = 1
    """DEPRECATED: compute Rsubt diagnostics every N iterations (use diag_interval_steps instead)"""
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
    """use deterministic (mean) actions for evaluation"""

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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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
        obs_noise_relative: bool = True,
    ):
        self.rng = np.random.default_rng(seed)
        self.obs_noise_std = obs_noise_std
        self.obs_dropout_p = obs_dropout_p
        self.action_delay = action_delay
        self.reward_scale = reward_scale
        self.obs_noise_relative = obs_noise_relative
        # Running stats for relative noise (per-dimension)
        self._obs_mean = None
        self._obs_var = None
        self._obs_count = 0
        
        # Action delay queue (per-env)
        if action_delay > 0:
            self.action_queue = [None] * action_delay
        else:
            self.action_queue = None
    
    def transform_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation noise and dropout."""
        if self.obs_noise_std > 0:
            # Match ObsNoiseWrapper: relative noise uses running per-dimension std.
            if self.obs_noise_relative:
                if self._obs_mean is None:
                    self._obs_mean = np.zeros(obs.shape[1:], dtype=np.float32)
                    self._obs_var = np.ones(obs.shape[1:], dtype=np.float32)
                # Update running stats per step using batch mean (good approximation for vector env)
                batch = obs.astype(np.float32)
                self._obs_count += 1
                delta = batch.mean(axis=0) - self._obs_mean
                self._obs_mean += delta / max(self._obs_count, 1)
                self._obs_var += delta * (batch.mean(axis=0) - self._obs_mean) - self._obs_var / max(self._obs_count, 1)
                obs_std = np.sqrt(np.maximum(self._obs_var, 1e-8))
                noise_std = self.obs_noise_std * obs_std
            else:
                noise_std = self.obs_noise_std
            noise = self.rng.normal(0, noise_std, size=obs.shape).astype(obs.dtype)
            obs = obs + noise
        
        if self.obs_dropout_p > 0:
            mask = self.rng.random(obs.shape) > self.obs_dropout_p
            obs = obs * mask.astype(obs.dtype)
        
        return obs
    
    def transform_reward(self, reward: np.ndarray) -> np.ndarray:
        """Apply reward scaling."""
        if self.reward_scale != 1.0:
            reward = reward * self.reward_scale
        return reward
    
    def delay_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action delay (FIFO queue)."""
        if self.action_queue is None:
            return action
        
        # Initialize queue with zeros if first action
        if self.action_queue[0] is None:
            for i in range(len(self.action_queue)):
                self.action_queue[i] = np.zeros_like(action)
        
        # Get delayed action and push current action
        delayed_action = self.action_queue[0]
        self.action_queue = self.action_queue[1:] + [action.copy()]
        return delayed_action


class RunningMeanStd:
    """Minimal running mean/std (like Gymnasium NormalizeObservation)."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float32)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean.astype(np.float32)
        self.var = new_var.astype(np.float32)
        self.count = float(tot_count)


class RewardNormalizer:
    """
    Reward normalization (like Gymnasium NormalizeReward) using running std of discounted returns.
    """

    def __init__(self, num_envs: int, gamma: float, epsilon: float = 1e-8):
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.returns = np.zeros((num_envs,), dtype=np.float32)
        self.ret_rms = RunningMeanStd((), epsilon=1e-4)

    def normalize(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        self.returns = self.returns * self.gamma + reward
        # Update stats on returns
        self.ret_rms.update(self.returns)
        # Normalize reward by std of returns
        std = float(np.sqrt(self.ret_rms.var + self.epsilon))
        out = reward / std
        # Reset returns on episode end
        self.returns *= (1.0 - done)
        return out


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # Critic with separate layers for activation extraction
        self.critic_fc1 = layer_init(nn.Linear(obs_dim, 64))
        self.critic_fc2 = layer_init(nn.Linear(64, 64))
        self.critic_out = layer_init(nn.Linear(64, 1), std=1.0)
        
        # Actor with separate layers for activation extraction
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, 64))
        self.actor_fc2 = layer_init(nn.Linear(64, 64))
        self.actor_out = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # For backward compatibility with actor_mean
        self.actor_mean = nn.Sequential(
            self.actor_fc1, nn.Tanh(),
            self.actor_fc2, nn.Tanh(),
            self.actor_out
        )
        self.critic = nn.Sequential(
            self.critic_fc1, nn.Tanh(),
            self.critic_fc2, nn.Tanh(),
            self.critic_out
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def get_hidden_activations(self, x):
        """Get hidden layer activations for representation rank computation."""
        actor_h1 = torch.tanh(self.actor_fc1(x))
        actor_h2 = torch.tanh(self.actor_fc2(actor_h1))
        critic_h1 = torch.tanh(self.critic_fc1(x))
        critic_h2 = torch.tanh(self.critic_fc2(critic_h1))
        return actor_h2, critic_h2


def evaluate_agent(
    agent: Agent,
    envpool_id: str,
    backend: str,
    num_episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool = True,
    obs_rms: RunningMeanStd | None = None,
    obs_clip: float = 10.0,
    stress: StressTransforms | None = None,
    reward_scale: float = 1.0,
) -> dict:
    """Run periodic evaluation using EnvPool environment (same as training).
    
    Uses EnvPool with num_envs=1 to ensure consistent observation shapes with training.
    """
    # Create EnvPool env with num_envs=1 for evaluation
    is_dmc = (backend == "dm")
    
    if is_dmc:
        eval_env = envpool.make_dm(envpool_id, num_envs=1, seed=seed)
        action_spec = eval_env.action_spec()
        action_low, action_high = action_bounds_from_envpool_dm(action_spec)
    else:
        eval_env = envpool.make(envpool_id, env_type="gym", num_envs=1, seed=seed)
        action_low, action_high = action_bounds_from_envpool_gym(eval_env.action_space)
    
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
            obs_arr = obs[0]  # Shape: (obs_dim,)
            
            # Apply observation normalization (same as training)
            if obs_rms is not None:
                obs_arr = (obs_arr - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
                obs_arr = np.clip(obs_arr, -obs_clip, obs_clip)
            
            # Apply stress transforms if configured
            if stress is not None:
                obs_arr = stress.transform_obs(obs_arr.reshape(1, -1)).reshape(-1)
            
            obs_tensor = torch.Tensor(obs_arr).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                if deterministic:
                    action = agent.actor_mean(obs_tensor)
            
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
            
            # Eval return should reflect stress reward scaling if configured
            ep_return += float(reward) * float(reward_scale)
            ep_length += 1
            
            # Safety limit to prevent infinite loops
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
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Convert env_id to EnvPool format
    envpool_id = get_envpool_id(args.env_id)
    backend = get_envpool_backend(envpool_id)
    env_type = detect_env_type(envpool_id)
    
    # Run name uses EnvPool ID for traceability
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
    # Log original env_id for traceability
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
        # DMC returns TimeStep, we need to handle differently
        is_dmc = True
    else:
        envs = envpool.make(envpool_id, env_type="gym", num_envs=args.num_envs, seed=args.seed)
        is_dmc = False
    
    # Get observation and action dimensions
    if is_dmc:
        # DMC observation is a State; flatten with helper after reset to infer dim.
        action_spec = envs.action_spec()
        action_low, action_high = action_bounds_from_envpool_dm(action_spec)
        action_dim = int(np.prod(action_spec.shape))
        # Infer obs_dim from a reset
        ts0 = envs.reset()
        obs0 = flatten_dmc_observation(ts0.observation, args.num_envs)
        obs_dim = obs0.shape[1]
        # Reset again to start training cleanly
        timestep = envs.reset()
    else:
        action_low, action_high = action_bounds_from_envpool_gym(envs.action_space)
        action_dim = int(np.prod(envs.action_space.shape))
        obs_dim = int(np.prod(envs.observation_space.shape))

    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Match original CleanRL PPO continuous: NormalizeObservation + TransformObservation clip(-10,10)
    obs_rms = RunningMeanStd((obs_dim,))
    obs_clip = 10.0
    # Match NormalizeReward + TransformReward clip(-10,10)
    reward_norm = RewardNormalizer(args.num_envs, gamma=args.gamma)
    reward_clip = 10.0

    # Episode statistics tracker
    episode_stats = RecordEpisodeStatistics(args.num_envs)
    
    # Stress transforms (array-level)
    stress_transforms = StressTransforms(
        num_envs=args.num_envs,
        obs_noise_std=args.obs_noise_std,
        obs_dropout_p=args.obs_dropout_p,
        action_delay=args.action_delay,
        reward_scale=args.reward_scale,
        seed=args.seed,
        obs_noise_relative=True,
    )

    # Rsubt stability certificate setup
    rsubt_monitor = None
    if args.rsubt_monitor:
        from cleanrl_utils.rsubt import RsubtMonitor
        actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
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

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    last_eval_step = 0
    
    # Reset environment
    if is_dmc:
        # `timestep` already created above
        next_obs_raw = flatten_dmc_observation(timestep.observation, args.num_envs).astype(np.float32)
    else:
        next_obs = envs.reset()
        next_obs_raw = next_obs.reshape(args.num_envs, -1).astype(np.float32)

    # Observation normalization (like original wrappers), then stress transforms
    obs_rms.update(next_obs_raw)
    next_obs_proc = (next_obs_raw - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    next_obs_proc = np.clip(next_obs_proc, -obs_clip, obs_clip)
    next_obs_proc = stress_transforms.transform_obs(next_obs_proc)
    next_obs = torch.Tensor(next_obs_proc).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    episode_stats.reset()

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action
            action_np = action.cpu().numpy()
            action_np = np.clip(action_np, action_low, action_high)
            action_np = stress_transforms.delay_action(action_np)
            
            if is_dmc:
                timestep = envs.step(action_np)
                next_obs_np_raw = flatten_dmc_observation(timestep.observation, args.num_envs).astype(np.float32)
                reward_raw = timestep.reward.astype(np.float32)
                next_done_np = timestep.last().astype(np.float32)
            else:
                next_obs_np, reward, done, info = envs.step(action_np)
                next_obs_np_raw = next_obs_np.reshape(args.num_envs, -1).astype(np.float32)
                reward_raw = np.asarray(reward, dtype=np.float32)
                next_done_np = done.astype(np.float32)
            
            # Observation normalization + clip, then stress transforms
            obs_rms.update(next_obs_np_raw)
            next_obs_np = (next_obs_np_raw - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
            next_obs_np = np.clip(next_obs_np, -obs_clip, obs_clip)
            next_obs_np = stress_transforms.transform_obs(next_obs_np)

            # Reward normalization + clip (like original), then stress reward scaling (outermost wrapper)
            reward_normed = reward_norm.normalize(reward_raw, next_done_np)
            reward_normed = np.clip(reward_normed, -reward_clip, reward_clip)
            reward = stress_transforms.transform_reward(reward_normed)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)
            
            # Track episode statistics
            ep_infos = episode_stats.update(reward_raw, next_done_np.astype(bool))
            for info in ep_infos:
                if info is not None:
                    ep_return = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    print(f"global_step={global_step}, episodic_return={ep_return}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)

        # Bootstrap value if not done
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

        # Flatten the batch
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # Rsubt: snapshot actor params before optimization
        if rsubt_monitor is not None:
            actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
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
                
                # Compute gradient norms before clipping
                actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
                critic_params = list(agent.critic.parameters())
                grad_norm_actor = sum(p.grad.norm().item() ** 2 for p in actor_params if p.grad is not None) ** 0.5
                grad_norm_critic = sum(p.grad.norm().item() ** 2 for p in critic_params if p.grad is not None) ** 0.5
                
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Rsubt: record update and compute certificate
        if rsubt_monitor is not None:
            actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
            rsubt_monitor.update(actor_params, global_step)
            rsubt_metrics = rsubt_monitor.maybe_compute(global_step)
            if rsubt_metrics is not None:
                rsubt_monitor.log_to_tensorboard(writer, global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TensorBoard logging - compatible with scoring suite contract
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # Baseline diagnostics: gradient norms
        writer.add_scalar("baseline/grad_norm_actor", grad_norm_actor, global_step)
        writer.add_scalar("baseline/grad_norm_critic", grad_norm_critic, global_step)
        
        # Representation rank diagnostics
        if args.rsubt_monitor and iteration % 10 == 0:
            from cleanrl_utils.rsubt.reprank import compute_effective_rank, compute_stable_rank
            with torch.no_grad():
                sample_obs = b_obs[:256]
                actor_hidden, critic_hidden = agent.get_hidden_activations(sample_obs)
                actor_effrank = compute_effective_rank(actor_hidden)
                actor_stablerank = compute_stable_rank(actor_hidden)
                critic_effrank = compute_effective_rank(critic_hidden)
                critic_stablerank = compute_stable_rank(critic_hidden)
            writer.add_scalar("reprank/actor_effrank", actor_effrank, global_step)
            writer.add_scalar("reprank/actor_stablerank", actor_stablerank, global_step)
            writer.add_scalar("reprank/critic_effrank", critic_effrank, global_step)
            writer.add_scalar("reprank/critic_stablerank", critic_stablerank, global_step)
        
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Periodic evaluation
        if args.eval_interval_steps > 0 and (global_step - last_eval_step) >= args.eval_interval_steps:
            eval_stats = evaluate_agent(
                agent,
                envpool_id,
                backend,
                args.eval_episodes,
                args.eval_seed,
                device,
                args.eval_deterministic,
                obs_rms=obs_rms,
                obs_clip=obs_clip,
                stress=stress_transforms,
                reward_scale=args.reward_scale,
            )
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)
            writer.add_scalar("eval/return_std", eval_stats["return_std"], global_step)
            writer.add_scalar("eval/length_mean", eval_stats["length_mean"], global_step)
            print(f"eval: global_step={global_step}, return_mean={eval_stats['return_mean']:.2f}, return_std={eval_stats['return_std']:.2f}")
            last_eval_step = global_step

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
