"""
Stress test wrappers for manufacturing failure modes in RL training.

These wrappers introduce various perturbations that are known to cause
instability in deep RL algorithms, useful for validating early warning
systems like RSUBT.

Usage:
    from cleanrl_utils.stress_wrappers import (
        ObsNoiseWrapper,
        ObsDropoutWrapper,
        ActionDelayWrapper,
        RewardScaleWrapper,
        DynamicsChangeWrapper,
        apply_stress_wrappers,
    )
    
    # Apply in make_env:
    env = gym.make(env_id)
    env = ObsNoiseWrapper(env, std=0.05)
    env = ActionDelayWrapper(env, delay_steps=2)
"""

from __future__ import annotations

from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class ObsNoiseWrapper(gym.ObservationWrapper):
    """
    Add Gaussian noise to observations.
    
    This simulates sensor noise and can destabilize learning,
    especially for algorithms relying on precise state estimation.
    
    Args:
        env: The environment to wrap.
        std: Standard deviation of noise (relative to obs std or absolute).
        relative: If True, std is relative to running obs std.
        seed: Random seed.
    """
    
    def __init__(
        self,
        env: gym.Env,
        std: float = 0.05,
        relative: bool = True,
        seed: int | None = None,
    ):
        super().__init__(env)
        self.std = std
        self.relative = relative
        self.rng = np.random.default_rng(seed)
        
        # Running stats for relative noise
        self._obs_mean = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._obs_var = np.ones(self.observation_space.shape, dtype=np.float32)
        self._obs_count = 0
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Add noise to observation."""
        # Update running stats
        self._obs_count += 1
        delta = observation - self._obs_mean
        self._obs_mean += delta / self._obs_count
        self._obs_var += delta * (observation - self._obs_mean) - self._obs_var / self._obs_count
        
        # Compute noise std
        if self.relative:
            obs_std = np.sqrt(np.maximum(self._obs_var, 1e-8))
            noise_std = self.std * obs_std
        else:
            noise_std = self.std
        
        noise = self.rng.normal(0, noise_std, observation.shape).astype(observation.dtype)
        return observation + noise


class ObsDropoutWrapper(gym.ObservationWrapper):
    """
    Randomly drop (zero out) observation dimensions.
    
    This simulates sensor failures and partial observability,
    which can cause sudden policy degradation.
    
    Args:
        env: The environment to wrap.
        p: Probability of dropping each dimension.
        seed: Random seed.
    """
    
    def __init__(
        self,
        env: gym.Env,
        p: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(env)
        self.p = p
        self.rng = np.random.default_rng(seed)
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Randomly zero out observation dimensions."""
        mask = self.rng.random(observation.shape) > self.p
        return observation * mask.astype(observation.dtype)


class ActionDelayWrapper(gym.Wrapper):
    """
    Delay action execution by a fixed number of steps.
    
    This simulates communication latency and is known to cause
    credit assignment issues and sudden collapses.
    
    Args:
        env: The environment to wrap.
        delay_steps: Number of steps to delay actions.
    """
    
    def __init__(
        self,
        env: gym.Env,
        delay_steps: int = 1,
    ):
        super().__init__(env)
        self.delay_steps = delay_steps
        self._action_queue: deque = deque(maxlen=delay_steps + 1)
        self._default_action = self._get_default_action()
    
    def _get_default_action(self) -> np.ndarray:
        """Get a default action (zeros or sample)."""
        if isinstance(self.action_space, gym.spaces.Box):
            # Use zeros for continuous
            return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return 0
        else:
            return self.action_space.sample()
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset environment and action queue."""
        obs, info = self.env.reset(**kwargs)
        self._action_queue.clear()
        # Fill queue with default actions
        for _ in range(self.delay_steps):
            self._action_queue.append(self._default_action)
        return obs, info
    
    def step(self, action: Any) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Step with delayed action."""
        # Add current action to queue
        self._action_queue.append(action)
        # Execute oldest action
        delayed_action = self._action_queue[0]
        return self.env.step(delayed_action)


class RewardScaleWrapper(gym.RewardWrapper):
    """
    Scale rewards by a constant factor.
    
    Extreme scaling (very high or very low) can break value function
    learning and cause instability.
    
    Args:
        env: The environment to wrap.
        scale: Reward multiplier.
    """
    
    def __init__(
        self,
        env: gym.Env,
        scale: float = 1.0,
    ):
        super().__init__(env)
        self.scale = scale
    
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Scale the reward."""
        return reward * self.scale


class DynamicsChangeWrapper(gym.Wrapper):
    """
    Change environment dynamics mid-training.
    
    At a specified step, modifies physics parameters (mass, friction, etc.)
    to simulate domain shift. This tests robustness and can cause sudden
    performance drops.
    
    Currently supports MuJoCo environments.
    
    Args:
        env: The environment to wrap.
        change_step: Global step at which to apply the change.
        mass_scale: Factor to multiply body masses.
        friction_scale: Factor to multiply friction coefficients.
        gravity_scale: Factor to multiply gravity.
    """
    
    def __init__(
        self,
        env: gym.Env,
        change_step: int = 500000,
        mass_scale: float = 1.0,
        friction_scale: float = 1.0,
        gravity_scale: float = 1.0,
    ):
        super().__init__(env)
        self.change_step = change_step
        self.mass_scale = mass_scale
        self.friction_scale = friction_scale
        self.gravity_scale = gravity_scale
        
        self._step_count = 0
        self._changed = False
        self._original_params: dict | None = None
    
    def _save_original_params(self) -> None:
        """Save original physics parameters."""
        try:
            model = self.unwrapped.model
            self._original_params = {
                "body_mass": model.body_mass.copy(),
                "geom_friction": model.geom_friction.copy(),
                "opt_gravity": model.opt.gravity.copy(),
            }
        except AttributeError:
            # Not a MuJoCo environment
            self._original_params = None
    
    def _apply_changes(self) -> None:
        """Apply physics parameter changes."""
        if self._original_params is None:
            return
        
        try:
            model = self.unwrapped.model
            
            if self.mass_scale != 1.0:
                model.body_mass[:] = self._original_params["body_mass"] * self.mass_scale
            
            if self.friction_scale != 1.0:
                model.geom_friction[:] = self._original_params["geom_friction"] * self.friction_scale
            
            if self.gravity_scale != 1.0:
                model.opt.gravity[:] = self._original_params["opt_gravity"] * self.gravity_scale
            
            self._changed = True
            print(f"DynamicsChangeWrapper: Applied changes at step {self._step_count}")
        except AttributeError:
            pass
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset environment."""
        if self._original_params is None:
            self._save_original_params()
        return self.env.reset(**kwargs)
    
    def step(self, action: Any) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Step and possibly apply dynamics change."""
        self._step_count += 1
        
        if not self._changed and self._step_count >= self.change_step:
            self._apply_changes()
        
        return self.env.step(action)
    
    def set_global_step(self, step: int) -> None:
        """Set the global step count (for coordination with training loop)."""
        self._step_count = step


# Stress presets
STRESS_PRESETS = {
    # Hyperparameter stress (applied via CLI args, not wrappers)
    "ppo_value_domination": {
        "description": "High vf_coef, low clip_coef, more epochs",
        "cli_args": {
            "vf_coef": 4.0,
            "clip_coef": 0.05,
            "update_epochs": 20,
        },
    },
    "ppo_lr_blowup": {
        "description": "High learning rate, more epochs, fewer minibatches",
        "cli_args": {
            "learning_rate": 3e-3,
            "update_epochs": 20,
            "num_minibatches": 4,
        },
    },
    "sac_critic_unstable": {
        "description": "High critic LR, fast target updates",
        "cli_args": {
            "q_lr": 3e-3,
            "tau": 0.05,
        },
    },
    "td3_fast_actor": {
        "description": "High actor LR, frequent policy updates",
        "cli_args": {
            "learning_rate": 1e-3,
            "policy_frequency": 1,
        },
    },
    
    # Environment stress (applied via wrappers)
    "obs_noise": {
        "description": "Gaussian observation noise",
        "wrappers": [
            {"type": "ObsNoiseWrapper", "std": 0.05, "relative": True},
        ],
    },
    "obs_noise_heavy": {
        "description": "Heavy Gaussian observation noise",
        "wrappers": [
            {"type": "ObsNoiseWrapper", "std": 0.15, "relative": True},
        ],
    },
    "obs_dropout": {
        "description": "Random observation dropout",
        "wrappers": [
            {"type": "ObsDropoutWrapper", "p": 0.1},
        ],
    },
    "obs_dropout_heavy": {
        "description": "Heavy observation dropout",
        "wrappers": [
            {"type": "ObsDropoutWrapper", "p": 0.3},
        ],
    },
    "action_delay_1": {
        "description": "1-step action delay",
        "wrappers": [
            {"type": "ActionDelayWrapper", "delay_steps": 1},
        ],
    },
    "action_delay_2": {
        "description": "2-step action delay",
        "wrappers": [
            {"type": "ActionDelayWrapper", "delay_steps": 2},
        ],
    },
    "action_delay_4": {
        "description": "4-step action delay",
        "wrappers": [
            {"type": "ActionDelayWrapper", "delay_steps": 4},
        ],
    },
    "reward_scale_high": {
        "description": "10x reward scaling",
        "wrappers": [
            {"type": "RewardScaleWrapper", "scale": 10.0},
        ],
    },
    "reward_scale_low": {
        "description": "0.1x reward scaling",
        "wrappers": [
            {"type": "RewardScaleWrapper", "scale": 0.1},
        ],
    },
    "dynamics_mass_up": {
        "description": "Increase body mass by 50% at 40% training",
        "wrappers": [
            {"type": "DynamicsChangeWrapper", "change_step": 400000, "mass_scale": 1.5},
        ],
    },
    "dynamics_mass_down": {
        "description": "Decrease body mass by 20% at 40% training",
        "wrappers": [
            {"type": "DynamicsChangeWrapper", "change_step": 400000, "mass_scale": 0.8},
        ],
    },
    "dynamics_friction": {
        "description": "Reduce friction by 30% at 40% training",
        "wrappers": [
            {"type": "DynamicsChangeWrapper", "change_step": 400000, "friction_scale": 0.7},
        ],
    },
    
    # Combined stress
    "combined_obs": {
        "description": "Noise + dropout combination",
        "wrappers": [
            {"type": "ObsNoiseWrapper", "std": 0.03, "relative": True},
            {"type": "ObsDropoutWrapper", "p": 0.05},
        ],
    },
    "combined_latency": {
        "description": "Noise + action delay",
        "wrappers": [
            {"type": "ObsNoiseWrapper", "std": 0.03, "relative": True},
            {"type": "ActionDelayWrapper", "delay_steps": 1},
        ],
    },
}


def get_wrapper_class(wrapper_type: str) -> type:
    """Get wrapper class by name."""
    mapping = {
        "ObsNoiseWrapper": ObsNoiseWrapper,
        "ObsDropoutWrapper": ObsDropoutWrapper,
        "ActionDelayWrapper": ActionDelayWrapper,
        "RewardScaleWrapper": RewardScaleWrapper,
        "DynamicsChangeWrapper": DynamicsChangeWrapper,
    }
    return mapping[wrapper_type]


def apply_stress_wrappers(
    env: gym.Env,
    preset: str | None = None,
    wrappers_config: list[dict] | None = None,
    seed: int | None = None,
) -> gym.Env:
    """
    Apply stress wrappers to an environment.
    
    Args:
        env: Base environment.
        preset: Name of a stress preset (see STRESS_PRESETS).
        wrappers_config: Custom list of wrapper configs.
        seed: Random seed for stochastic wrappers.
    
    Returns:
        Wrapped environment.
    """
    configs = []
    
    if preset is not None:
        if preset not in STRESS_PRESETS:
            raise ValueError(f"Unknown stress preset: {preset}. Available: {list(STRESS_PRESETS.keys())}")
        preset_config = STRESS_PRESETS[preset]
        if "wrappers" in preset_config:
            configs.extend(preset_config["wrappers"])
    
    if wrappers_config is not None:
        configs.extend(wrappers_config)
    
    for config in configs:
        wrapper_type = config.pop("type")
        wrapper_class = get_wrapper_class(wrapper_type)
        
        # Add seed if the wrapper accepts it
        if seed is not None and "seed" in wrapper_class.__init__.__code__.co_varnames:
            config["seed"] = seed
        
        env = wrapper_class(env, **config)
        config["type"] = wrapper_type  # Restore for potential reuse
    
    return env


def get_stress_preset_cli_args(preset: str) -> dict:
    """
    Get CLI args for a hyperparameter stress preset.
    
    Returns empty dict for environment-only presets.
    """
    if preset not in STRESS_PRESETS:
        return {}
    return STRESS_PRESETS[preset].get("cli_args", {})


def list_stress_presets() -> list[str]:
    """List available stress presets."""
    return list(STRESS_PRESETS.keys())
