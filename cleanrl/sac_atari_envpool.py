# EnvPool-native SAC for Atari (discrete action space)
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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.envpool_registry import get_envpool_id, detect_env_type
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
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment (EnvPool format)"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    num_envs: int = 8
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 20000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""

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
    """DEPRECATED: compute Rsubt diagnostics every N actor updates (use diag_interval_steps instead)"""
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

    # Stress test arguments (array-level transforms)
    stress_preset: str = ""
    """stress test preset name (see cleanrl_utils.stress_wrappers.STRESS_PRESETS)"""
    obs_noise_std: float = 0.0
    """observation noise std (0 to disable, applied after normalization)"""
    action_delay: int = 0
    """action delay in steps (0 to disable)"""
    reward_scale: float = 1.0
    """reward scaling factor"""


class RecordEpisodeStatistics:
    """Track episode statistics for EnvPool Atari environments with episodic life."""
    
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
    
    def reset(self):
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
    
    def update(self, rewards: np.ndarray, terminated: np.ndarray, lives: np.ndarray):
        """Update statistics. For Atari with episodic life, only log on true game over (lives==0)."""
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        infos = []
        for i, (term, live) in enumerate(zip(terminated, lives)):
            if term and live == 0:
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
    """Array-level stress transforms for EnvPool Atari environments."""
    
    def __init__(
        self,
        num_envs: int,
        obs_noise_std: float = 0.0,
        action_delay: int = 0,
        reward_scale: float = 1.0,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.obs_noise_std = obs_noise_std
        self.action_delay = action_delay
        self.reward_scale = reward_scale
        
        if action_delay > 0:
            self.action_queue = [None] * action_delay
        else:
            self.action_queue = None
    
    def transform_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation noise to normalized observations [0, 1]."""
        if self.obs_noise_std > 0:
            noise = self.rng.normal(0, self.obs_noise_std, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0, 1)
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
    """Simple replay buffer for Atari observations."""
    
    def __init__(self, buffer_size: int, obs_shape: tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Use uint8 for observations to save memory
        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
    
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
            torch.FloatTensor(self.observations[idxs] / 255.0).to(self.device),
            torch.FloatTensor(self.next_observations[idxs] / 255.0).to(self.device),
            torch.LongTensor(self.actions[idxs]).to(self.device),
            torch.FloatTensor(self.rewards[idxs]).to(self.device),
            torch.FloatTensor(self.dones[idxs]).to(self.device),
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, num_actions: int):
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
        self.q_head = layer_init(nn.Linear(512, num_actions), std=1.0)

    def forward(self, x):
        return self.q_head(self.network(x))


class Actor(nn.Module):
    def __init__(self, num_actions: int):
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
        self.actor_head = layer_init(nn.Linear(512, num_actions), std=0.01)

    def forward(self, x):
        return self.actor_head(self.network(x))

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


def evaluate_agent(
    actor: Actor,
    env_id: str,
    num_episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool = True,
) -> dict:
    """Run periodic evaluation using EnvPool (no episodic life for clean eval)."""
    env = envpool.make(
        env_id, env_type="gym", num_envs=1, seed=seed,
        episodic_life=False, reward_clip=False,
    )
    
    returns = []
    lengths = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        
        while not done:
            obs_tensor = torch.FloatTensor(obs / 255.0).to(device)
            with torch.no_grad():
                if deterministic:
                    logits = actor(obs_tensor)
                    action = logits.argmax(dim=1).cpu().numpy()
                else:
                    action, _, _ = actor.get_action(obs_tensor)
                    action = action.cpu().numpy()
            
            obs, reward, done, info = env.step(action)
            done = done[0]
            ep_return += reward[0]
            ep_length += 1
        
        returns.append(ep_return)
        lengths.append(ep_length)
    
    env.close()
    
    return {
        "return_mean": np.mean(returns),
        "return_std": np.std(returns),
        "length_mean": np.mean(lengths),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    apply_stress_preset_to_args(args)
    
    # Convert env_id to EnvPool format if needed
    envpool_id = get_envpool_id(args.env_id)
    
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

    # Create EnvPool Atari environment
    envs = envpool.make(
        envpool_id,
        env_type="gym",
        num_envs=args.num_envs,
        seed=args.seed,
        episodic_life=True,
        reward_clip=True,
    )
    
    num_actions = envs.action_space.n
    obs_shape = envs.observation_space.shape  # (4, 84, 84)

    # Initialize networks
    actor = Actor(num_actions).to(device)
    qf1 = SoftQNetwork(num_actions).to(device)
    qf2 = SoftQNetwork(num_actions).to(device)
    qf1_target = SoftQNetwork(num_actions).to(device)
    qf2_target = SoftQNetwork(num_actions).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * np.log(1.0 / num_actions)
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
        action_delay=args.action_delay,
        reward_scale=args.reward_scale,
        seed=args.seed,
    )

    # Replay buffer
    rb = ReplayBuffer(args.buffer_size, obs_shape, device)

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
        )

    start_time = time.time()
    last_eval_step = 0
    grad_norm_actor = 0.0
    actor_loss = torch.tensor(0.0)

    # Reset environment
    obs = envs.reset()
    episode_stats.reset()

    for global_step in range(args.total_timesteps):
        # Action logic
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(args.num_envs)])
        else:
            obs_float = (obs / 255.0).astype(np.float32)
            obs_float = stress_transforms.transform_obs(obs_float)
            obs_tensor = torch.FloatTensor(obs_float).to(device)
            actions, _, _ = actor.get_action(obs_tensor)
            actions = actions.cpu().numpy()
        
        actions = stress_transforms.delay_action(actions)
        
        # Execute action
        next_obs, rewards, dones, infos = envs.step(actions)
        
        # Apply stress transforms
        rewards = stress_transforms.transform_reward(rewards)
        
        # Track episode statistics (with episodic life handling)
        lives = infos.get("lives", np.zeros(args.num_envs))
        terminated = infos.get("terminated", dones).astype(np.bool_)
        # For logging episodic return, only log on true game over (terminated & lives==0)
        ep_infos = episode_stats.update(infos.get("reward", rewards), terminated, lives)
        for info in ep_infos:
            if info is not None:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        # IMPORTANT: mimic EpisodicLifeEnv semantics for learning => use `dones` (life loss ends episode for bootstrapping).
        rb.add(obs, next_obs, actions, rewards, dones.astype(np.float32))
        obs = next_obs

        # Training
        if global_step > args.learning_starts and global_step % args.update_frequency == 0:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)
            # Apply obs noise stress in training batches (normalized space)
            if args.obs_noise_std > 0:
                b_obs = torch.clamp(b_obs + torch.randn_like(b_obs) * args.obs_noise_std, 0.0, 1.0)
                b_next_obs = torch.clamp(b_next_obs + torch.randn_like(b_next_obs) * args.obs_noise_std, 0.0, 1.0)
            
            with torch.no_grad():
                _, next_state_log_pi, next_state_action_probs = actor.get_action(b_next_obs)
                qf1_next_target = qf1_target(b_next_obs)
                qf2_next_target = qf2_target(b_next_obs)
                min_qf_next_target = next_state_action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )
                min_qf_next_target = min_qf_next_target.sum(dim=1)
                next_q_value = b_rewards + (1 - b_dones) * args.gamma * min_qf_next_target

            qf1_a_values = qf1(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze()
            qf2_a_values = qf2(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze()
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Actor update
            # Rsubt: snapshot actor params before update
            if rsubt_monitor is not None:
                rsubt_monitor.snapshot_params(list(actor.parameters()))

            _, log_pi, action_probs = actor.get_action(b_obs)
            with torch.no_grad():
                qf1_values = qf1(b_obs)
                qf2_values = qf2(b_obs)
                min_qf_values = torch.min(qf1_values, qf2_values)
            
            actor_loss = (action_probs * (alpha * log_pi - min_qf_values)).sum(1).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            grad_norm_actor = sum(p.grad.norm().item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
            actor_optimizer.step()

            # Rsubt: record update and compute certificate
            if rsubt_monitor is not None:
                rsubt_monitor.update(list(actor.parameters()), global_step)
                rsubt_metrics = rsubt_monitor.maybe_compute(global_step)
                if rsubt_metrics is not None:
                    rsubt_monitor.log_to_tensorboard(writer, global_step)

            if args.autotune:
                with torch.no_grad():
                    _, log_pi, action_probs = actor.get_action(b_obs)
                alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).sum(1).mean()
                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

        # Update target networks
        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        # Logging
        if global_step > args.learning_starts and global_step % 100 == 0:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            writer.add_scalar("baseline/grad_norm_actor", grad_norm_actor, global_step)
            
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Periodic evaluation
        if args.eval_interval_steps > 0 and (global_step - last_eval_step) >= args.eval_interval_steps:
            eval_stats = evaluate_agent(actor, envpool_id, args.eval_episodes, args.eval_seed, device, args.eval_deterministic)
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)
            writer.add_scalar("eval/return_std", eval_stats["return_std"], global_step)
            writer.add_scalar("eval/length_mean", eval_stats["length_mean"], global_step)
            print(f"eval: global_step={global_step}, return_mean={eval_stats['return_mean']:.2f}")
            last_eval_step = global_step

    envs.close()
    writer.close()
