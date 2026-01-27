"""
EnvPool vector-env helpers used by the EnvPool-native CleanRL scripts.

These utilities focus on **correctness** and matching the original CleanRL
semantics as closely as possible, especially for:

- DMC observations returned as `envpool.python.data.State` (namedtuple-like)
- Separating termination vs time-limit truncation for Gym-style EnvPool envs
- Action bounds handling as vectors (per-dimension), not scalars
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def flatten_dmc_observation(obs: Any, num_envs: int) -> np.ndarray:
    """
    Flatten an EnvPool DMC observation into a 2D array of shape (num_envs, obs_dim).

    EnvPool DMC uses `dm_env.TimeStep` with `observation` often being an
    `envpool.python.data.State` (namedtuple-like) containing both metadata
    fields (e.g. env_id, players) and numeric observation arrays (e.g. position, velocity).

    We concatenate only numeric ndarray fields whose first dimension matches `num_envs`.
    """
    if isinstance(obs, np.ndarray):
        return obs.reshape(num_envs, -1)

    # Mapping-like
    if hasattr(obs, "items"):
        parts: list[np.ndarray] = []
        for _, v in obs.items():
            if isinstance(v, np.ndarray) and v.shape[0] == num_envs:
                parts.append(v.reshape(num_envs, -1))
        if not parts:
            raise ValueError("DMC observation mapping contained no numeric array fields")
        return np.concatenate(parts, axis=1)

    # Namedtuple-like (EnvPool State)
    if hasattr(obs, "_asdict"):
        d = obs._asdict()
        parts = []
        for _, v in d.items():
            if isinstance(v, np.ndarray) and v.shape[0] == num_envs and v.dtype.kind in {"f", "c", "b"}:
                parts.append(v.reshape(num_envs, -1))
        if not parts:
            raise ValueError("DMC State contained no numeric float/bool array fields")
        return np.concatenate(parts, axis=1)

    raise TypeError(f"Unsupported DMC observation type: {type(obs)}")


def info_time_limit_truncated(info: dict[str, Any], num_envs: int) -> np.ndarray:
    """
    Return a boolean array (num_envs,) indicating time-limit truncation for EnvPool gym envs.

    EnvPool gym envs commonly provide `TimeLimit.truncated` in the `info` dict.
    If missing, returns all-False.
    """
    tl = info.get("TimeLimit.truncated", None)
    if tl is None:
        return np.zeros((num_envs,), dtype=bool)
    tl = np.asarray(tl).astype(bool)
    if tl.shape == ():
        tl = np.repeat(tl, num_envs)
    return tl.reshape(num_envs)


@dataclass(frozen=True)
class DoneSplit:
    terminations: np.ndarray  # True terminal (non-time-limit)
    truncations: np.ndarray  # True time-limit truncation


def split_terminations_truncations(done: np.ndarray, info: dict[str, Any]) -> DoneSplit:
    """
    Split EnvPool gym `done` into terminations and time-limit truncations.

    Matches CleanRL's usual practice of treating time-limit truncations as non-terminal
    for off-policy bootstrapping (i.e. done=False for truncations).
    """
    done = np.asarray(done).astype(bool)
    num_envs = int(done.shape[0])
    trunc = info_time_limit_truncated(info, num_envs)
    term = done & (~trunc)
    return DoneSplit(terminations=term, truncations=trunc)


def action_bounds_from_envpool_gym(action_space) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) as float32 vectors from a gym-style action_space."""
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return low, high


def action_bounds_from_envpool_dm(action_spec) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) as float32 vectors from a dm_env action_spec."""
    low = np.asarray(action_spec.minimum, dtype=np.float32)
    high = np.asarray(action_spec.maximum, dtype=np.float32)
    return low, high


def action_scale_bias(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dimension action scale/bias used by CleanRL TD3/SAC implementations.
    """
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0
    return scale, bias

