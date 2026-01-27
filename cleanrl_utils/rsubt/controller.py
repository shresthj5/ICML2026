"""
RiskController: Hyperparameter controller based on Rsubt alarm state.

Implements risk-gated modifications to learning rate, PPO epochs/clip,
or SAC actor update frequency when stability certificate enters
yellow or red states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cleanrl_utils.rsubt.certificate import AlarmState


@dataclass
class PPOHyperparams:
    """Hyperparameters that can be modified for PPO."""
    learning_rate: float
    update_epochs: int
    clip_coef: float


@dataclass
class SACHyperparams:
    """Hyperparameters that can be modified for SAC."""
    actor_lr: float
    policy_frequency: int  # actor update every N critic updates
    skip_actor: bool = False  # skip actor updates entirely this step


@dataclass
class TD3Hyperparams:
    """Hyperparameters that can be modified for TD3."""
    actor_lr: float
    policy_frequency: int  # actor update every N critic updates
    skip_actor: bool = False  # skip actor updates entirely this step
    exploration_noise: float = 0.1  # can reduce noise when unstable


class RiskController:
    """
    Risk-gated hyperparameter controller.
    
    Adjusts hyperparameters based on alarm state to mitigate instability:
    - GREEN: Use base hyperparameters.
    - YELLOW: Reduce learning rate, reduce PPO epochs.
    - RED: Aggressive reduction, potentially skip actor updates.
    
    Args:
        algorithm: Either "ppo", "sac", or "td3".
        base_lr: Base learning rate (for PPO shared, for SAC/TD3 actor).
        base_epochs: Base PPO epochs (only for PPO).
        base_clip: Base PPO clip coefficient (only for PPO).
        base_policy_freq: Base SAC/TD3 policy frequency (only for SAC/TD3).
        base_exploration_noise: Base TD3 exploration noise (only for TD3).
        yellow_lr_mult: LR multiplier for yellow state.
        red_lr_mult: LR multiplier for red state.
        yellow_epoch_reduction: Epochs to subtract in yellow.
        red_clip_override: Clip coefficient in red state.
        red_skip_steps: Number of iterations to skip actor in red.
        min_lr_mult: Minimum LR multiplier (floor).
        min_epochs: Minimum PPO epochs.
    """
    
    def __init__(
        self,
        algorithm: str = "ppo",
        base_lr: float = 3e-4,
        base_epochs: int = 10,
        base_clip: float = 0.2,
        base_policy_freq: int = 2,
        base_exploration_noise: float = 0.1,
        yellow_lr_mult: float = 0.7,
        red_lr_mult: float = 0.3,
        yellow_epoch_reduction: int = 2,
        red_clip_override: float = 0.05,
        red_skip_steps: int = 1,
        min_lr_mult: float = 0.1,
        min_epochs: int = 2,
    ):
        self.algorithm = algorithm.lower()
        self.base_lr = base_lr
        self.base_epochs = base_epochs
        self.base_clip = base_clip
        self.base_policy_freq = base_policy_freq
        self.base_exploration_noise = base_exploration_noise
        
        self.yellow_lr_mult = yellow_lr_mult
        self.red_lr_mult = red_lr_mult
        self.yellow_epoch_reduction = yellow_epoch_reduction
        self.red_clip_override = red_clip_override
        self.red_skip_steps = red_skip_steps
        self.min_lr_mult = min_lr_mult
        self.min_epochs = min_epochs
        
        # Track consecutive red states
        self._red_count = 0
        self._current_state = AlarmState.GREEN
    
    def get_hyperparams(self, state: AlarmState) -> PPOHyperparams | SACHyperparams | TD3Hyperparams:
        """
        Get modified hyperparameters based on alarm state.
        
        Args:
            state: Current alarm state.
        
        Returns:
            Modified hyperparameters object.
        """
        # Track state transitions
        if state == AlarmState.RED:
            self._red_count += 1
        else:
            self._red_count = 0
        self._current_state = state
        
        if self.algorithm == "ppo":
            return self._get_ppo_hyperparams(state)
        elif self.algorithm == "td3":
            return self._get_td3_hyperparams(state)
        else:
            return self._get_sac_hyperparams(state)
    
    def _get_ppo_hyperparams(self, state: AlarmState) -> PPOHyperparams:
        """Compute PPO hyperparameters for given state."""
        if state == AlarmState.GREEN:
            return PPOHyperparams(
                learning_rate=self.base_lr,
                update_epochs=self.base_epochs,
                clip_coef=self.base_clip,
            )
        
        elif state == AlarmState.YELLOW:
            lr = max(self.base_lr * self.yellow_lr_mult, 
                     self.base_lr * self.min_lr_mult)
            epochs = max(self.base_epochs - self.yellow_epoch_reduction, 
                        self.min_epochs)
            return PPOHyperparams(
                learning_rate=lr,
                update_epochs=epochs,
                clip_coef=self.base_clip,
            )
        
        else:  # RED
            lr = max(self.base_lr * self.red_lr_mult,
                     self.base_lr * self.min_lr_mult)
            epochs = max(self.min_epochs, 2)
            return PPOHyperparams(
                learning_rate=lr,
                update_epochs=epochs,
                clip_coef=self.red_clip_override,
            )
    
    def _get_sac_hyperparams(self, state: AlarmState) -> SACHyperparams:
        """Compute SAC hyperparameters for given state."""
        if state == AlarmState.GREEN:
            return SACHyperparams(
                actor_lr=self.base_lr,
                policy_frequency=self.base_policy_freq,
                skip_actor=False,
            )
        
        elif state == AlarmState.YELLOW:
            lr = max(self.base_lr * self.yellow_lr_mult,
                     self.base_lr * self.min_lr_mult)
            return SACHyperparams(
                actor_lr=lr,
                policy_frequency=self.base_policy_freq * 2,  # slower actor updates
                skip_actor=False,
            )
        
        else:  # RED
            lr = max(self.base_lr * self.red_lr_mult,
                     self.base_lr * self.min_lr_mult)
            # Skip actor for first N iterations in red
            skip = self._red_count <= self.red_skip_steps
            return SACHyperparams(
                actor_lr=lr,
                policy_frequency=self.base_policy_freq * 2,
                skip_actor=skip,
            )
    
    def _get_td3_hyperparams(self, state: AlarmState) -> TD3Hyperparams:
        """Compute TD3 hyperparameters for given state."""
        if state == AlarmState.GREEN:
            return TD3Hyperparams(
                actor_lr=self.base_lr,
                policy_frequency=self.base_policy_freq,
                skip_actor=False,
                exploration_noise=self.base_exploration_noise,
            )
        
        elif state == AlarmState.YELLOW:
            lr = max(self.base_lr * self.yellow_lr_mult,
                     self.base_lr * self.min_lr_mult)
            # Reduce exploration noise slightly when unstable
            noise = self.base_exploration_noise * 0.8
            return TD3Hyperparams(
                actor_lr=lr,
                policy_frequency=self.base_policy_freq * 2,  # slower actor updates
                skip_actor=False,
                exploration_noise=noise,
            )
        
        else:  # RED
            lr = max(self.base_lr * self.red_lr_mult,
                     self.base_lr * self.min_lr_mult)
            # Skip actor for first N iterations in red
            skip = self._red_count <= self.red_skip_steps
            # Further reduce noise in red state
            noise = self.base_exploration_noise * 0.5
            return TD3Hyperparams(
                actor_lr=lr,
                policy_frequency=self.base_policy_freq * 2,
                skip_actor=skip,
                exploration_noise=noise,
            )
    
    def reset(self) -> None:
        """Reset controller state."""
        self._red_count = 0
        self._current_state = AlarmState.GREEN
