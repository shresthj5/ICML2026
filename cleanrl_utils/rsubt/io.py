"""
IO utilities for Rsubt logging and TensorBoard integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.rsubt.certificate import AlarmState


# Metric name prefixes
RSUBT_PREFIX = "rsubt"
BASELINE_PREFIX = "baseline"
REPRANK_PREFIX = "reprank"


def log_rsubt_metrics(
    writer: "SummaryWriter",
    global_step: int,
    raw: float,
    ewma: float,
    state: AlarmState,
    shock: float,
    gap_ratio: float,
    update_norm: float = 0.0,
) -> None:
    """
    Log Rsubt certificate metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter.
        global_step: Current global step.
        raw: Raw Rsubt score.
        ewma: EWMA-smoothed Rsubt score.
        state: Current alarm state.
        shock: Directional shock value.
        gap_ratio: Gap ratio value.
        update_norm: Norm of the parameter update.
    """
    writer.add_scalar(f"{RSUBT_PREFIX}/raw", raw, global_step)
    writer.add_scalar(f"{RSUBT_PREFIX}/ewma", ewma, global_step)
    writer.add_scalar(f"{RSUBT_PREFIX}/state", int(state), global_step)
    writer.add_scalar(f"{RSUBT_PREFIX}/shock", shock, global_step)
    writer.add_scalar(f"{RSUBT_PREFIX}/gap_ratio", gap_ratio, global_step)
    writer.add_scalar(f"{RSUBT_PREFIX}/update_norm", update_norm, global_step)


def log_baseline_metrics(
    writer: "SummaryWriter",
    global_step: int,
    grad_norm_actor: float | None = None,
    grad_norm_critic: float | None = None,
    update_norm_actor: float | None = None,
) -> None:
    """
    Log baseline diagnostic metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter.
        global_step: Current global step.
        grad_norm_actor: Actor gradient norm.
        grad_norm_critic: Critic gradient norm.
        update_norm_actor: Actor update norm.
    """
    if grad_norm_actor is not None:
        writer.add_scalar(f"{BASELINE_PREFIX}/grad_norm_actor", grad_norm_actor, global_step)
    if grad_norm_critic is not None:
        writer.add_scalar(f"{BASELINE_PREFIX}/grad_norm_critic", grad_norm_critic, global_step)
    if update_norm_actor is not None:
        writer.add_scalar(f"{BASELINE_PREFIX}/update_norm_actor", update_norm_actor, global_step)


def log_reprank_metrics(
    writer: "SummaryWriter",
    global_step: int,
    actor_effrank: float | None = None,
    actor_stablerank: float | None = None,
    critic_effrank: float | None = None,
    critic_stablerank: float | None = None,
) -> None:
    """
    Log representation rank metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter.
        global_step: Current global step.
        actor_effrank: Actor effective rank.
        actor_stablerank: Actor stable rank.
        critic_effrank: Critic effective rank.
        critic_stablerank: Critic stable rank.
    """
    if actor_effrank is not None:
        writer.add_scalar(f"{REPRANK_PREFIX}/actor_effrank", actor_effrank, global_step)
    if actor_stablerank is not None:
        writer.add_scalar(f"{REPRANK_PREFIX}/actor_stablerank", actor_stablerank, global_step)
    if critic_effrank is not None:
        writer.add_scalar(f"{REPRANK_PREFIX}/critic_effrank", critic_effrank, global_step)
    if critic_stablerank is not None:
        writer.add_scalar(f"{REPRANK_PREFIX}/critic_stablerank", critic_stablerank, global_step)


def state_to_string(state: AlarmState) -> str:
    """Convert alarm state to string."""
    return ["green", "yellow", "red"][int(state)]


def string_to_state(s: str) -> AlarmState:
    """Convert string to alarm state."""
    mapping = {"green": AlarmState.GREEN, "yellow": AlarmState.YELLOW, "red": AlarmState.RED}
    return mapping.get(s.lower(), AlarmState.GREEN)
