"""
Online evaluation helper for periodic eval during training.

Provides a unified interface for running deterministic evaluations
at fixed intervals and logging to TensorBoard.

Usage in training scripts:
    from cleanrl_utils.evals.online_eval import OnlineEvaluator
    
    evaluator = OnlineEvaluator(
        make_env_fn=make_env,
        env_id=args.env_id,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        device=device,
        deterministic=args.eval_deterministic,
    )
    
    # In training loop:
    if global_step % args.eval_interval_steps == 0:
        eval_stats = evaluator.evaluate(agent)
        evaluator.log_to_tensorboard(writer, global_step, eval_stats)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Protocol

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class ActorProtocol(Protocol):
    """Protocol for actor models that can select actions."""
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class EvalStats:
    """Statistics from an evaluation run."""
    return_mean: float
    return_std: float
    return_min: float
    return_max: float
    length_mean: float
    length_std: float
    episode_returns: list[float]
    episode_lengths: list[int]


class OnlineEvaluator:
    """
    Online evaluator for periodic evaluation during training.
    
    Handles env creation, episode rollouts, and stats computation.
    
    Args:
        make_env_fn: Factory function to create an environment.
        env_id: Environment ID string.
        eval_episodes: Number of episodes to evaluate.
        eval_seed: Fixed seed for reproducibility.
        device: PyTorch device.
        deterministic: If True, use deterministic actions (mean for Gaussian policies).
        gamma: Discount factor (for envs that need it like PPO continuous).
        capture_video: Whether to record video (usually False for periodic eval).
    """
    
    def __init__(
        self,
        make_env_fn: Callable,
        env_id: str,
        eval_episodes: int = 10,
        eval_seed: int = 12345,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        gamma: float = 0.99,
        capture_video: bool = False,
        run_name: str = "eval",
    ):
        self.make_env_fn = make_env_fn
        self.env_id = env_id
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.gamma = gamma
        self.capture_video = capture_video
        self.run_name = run_name
        
        # Lazy init
        self._env = None
    
    def _get_env(self) -> gym.Env:
        """Get or create the evaluation environment."""
        if self._env is None:
            # Try different make_env signatures used across CleanRL
            try:
                # PPO continuous style: make_env(env_id, idx, capture_video, run_name, gamma)
                env_fn = self.make_env_fn(
                    self.env_id, 0, self.capture_video, self.run_name, self.gamma
                )
                self._env = env_fn()
            except TypeError:
                try:
                    # TD3/SAC style: make_env(env_id, seed, idx, capture_video, run_name)
                    env_fn = self.make_env_fn(
                        self.env_id, self.eval_seed, 0, self.capture_video, self.run_name
                    )
                    self._env = env_fn()
                except TypeError:
                    try:
                        # Atari style: make_env(env_id, idx, capture_video, run_name)
                        env_fn = self.make_env_fn(
                            self.env_id, 0, self.capture_video, self.run_name
                        )
                        self._env = env_fn()
                    except TypeError:
                        # Fallback: just env_id
                        self._env = gym.make(self.env_id)
        return self._env
    
    def evaluate_ppo(
        self,
        agent: nn.Module,
    ) -> EvalStats:
        """
        Evaluate a PPO-style agent (with get_action_and_value method).
        
        Args:
            agent: PPO agent with get_action_and_value(obs, action=None) method.
        
        Returns:
            EvalStats with episode statistics.
        """
        env = self._get_env()
        
        episode_returns = []
        episode_lengths = []
        
        for ep in range(self.eval_episodes):
            obs, _ = env.reset(seed=self.eval_seed + ep)
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.deterministic:
                        # For PPO with Gaussian policy, use mean action
                        if hasattr(agent, 'actor_mean'):
                            action = agent.actor_mean(obs_tensor)
                        else:
                            action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    else:
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)
                
                action_np = action.cpu().numpy().squeeze(0)
                
                # Handle discrete vs continuous
                if isinstance(env.action_space, gym.spaces.Discrete):
                    if action_np.ndim == 0:
                        action_np = int(action_np)
                    else:
                        action_np = int(action_np.argmax()) if action_np.ndim > 0 else int(action_np)
                else:
                    action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return EvalStats(
            return_mean=float(np.mean(episode_returns)),
            return_std=float(np.std(episode_returns)),
            return_min=float(np.min(episode_returns)),
            return_max=float(np.max(episode_returns)),
            length_mean=float(np.mean(episode_lengths)),
            length_std=float(np.std(episode_lengths)),
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
        )
    
    def evaluate_offpolicy(
        self,
        actor: nn.Module,
        exploration_noise: float = 0.0,
    ) -> EvalStats:
        """
        Evaluate an off-policy actor (TD3/SAC style with direct action output).
        
        Args:
            actor: Actor network that takes obs and returns actions.
            exploration_noise: Noise to add (0.0 for deterministic eval).
        
        Returns:
            EvalStats with episode statistics.
        """
        env = self._get_env()
        
        episode_returns = []
        episode_lengths = []
        
        for ep in range(self.eval_episodes):
            obs, _ = env.reset(seed=self.eval_seed + ep)
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action = actor(obs_tensor)
                    if exploration_noise > 0 and hasattr(actor, 'action_scale'):
                        action += torch.normal(0, actor.action_scale * exploration_noise, size=action.shape).to(self.device)
                
                action_np = action.cpu().numpy().squeeze(0)
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return EvalStats(
            return_mean=float(np.mean(episode_returns)),
            return_std=float(np.std(episode_returns)),
            return_min=float(np.min(episode_returns)),
            return_max=float(np.max(episode_returns)),
            length_mean=float(np.mean(episode_lengths)),
            length_std=float(np.std(episode_lengths)),
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
        )
    
    def evaluate_sac_discrete(
        self,
        actor: nn.Module,
    ) -> EvalStats:
        """
        Evaluate a discrete SAC actor (Atari style with get_action method).
        
        Args:
            actor: Actor with get_action(obs) returning (action, log_prob, action_probs).
        
        Returns:
            EvalStats with episode statistics.
        """
        env = self._get_env()
        
        episode_returns = []
        episode_lengths = []
        
        for ep in range(self.eval_episodes):
            obs, _ = env.reset(seed=self.eval_seed + ep)
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.deterministic:
                        # Use argmax of action probs for deterministic
                        if hasattr(actor, 'forward'):
                            logits = actor(obs_tensor / 255.0)
                            action = logits.argmax(dim=-1)
                        else:
                            action, _, action_probs = actor.get_action(obs_tensor)
                            action = action_probs.argmax(dim=-1)
                    else:
                        action, _, _ = actor.get_action(obs_tensor)
                
                action_np = action.cpu().numpy().squeeze(0)
                if isinstance(action_np, np.ndarray):
                    action_np = int(action_np.item()) if action_np.ndim == 0 else int(action_np[0])
                else:
                    action_np = int(action_np)
                
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return EvalStats(
            return_mean=float(np.mean(episode_returns)),
            return_std=float(np.std(episode_returns)),
            return_min=float(np.min(episode_returns)),
            return_max=float(np.max(episode_returns)),
            length_mean=float(np.mean(episode_lengths)),
            length_std=float(np.std(episode_lengths)),
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
        )
    
    def log_to_tensorboard(
        self,
        writer,
        global_step: int,
        stats: EvalStats,
        prefix: str = "eval",
    ) -> None:
        """
        Log evaluation statistics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter.
            global_step: Current global step.
            stats: EvalStats from evaluation.
            prefix: Prefix for metric names.
        """
        writer.add_scalar(f"{prefix}/return_mean", stats.return_mean, global_step)
        writer.add_scalar(f"{prefix}/return_std", stats.return_std, global_step)
        writer.add_scalar(f"{prefix}/return_min", stats.return_min, global_step)
        writer.add_scalar(f"{prefix}/return_max", stats.return_max, global_step)
        writer.add_scalar(f"{prefix}/length_mean", stats.length_mean, global_step)
    
    def close(self) -> None:
        """Close the evaluation environment."""
        if self._env is not None:
            self._env.close()
            self._env = None


def make_eval_env_simple(env_id: str, seed: int = 12345) -> gym.Env:
    """
    Create a simple eval environment without wrappers that affect rewards.
    
    Useful for getting raw, unscaled returns for collapse detection.
    """
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
