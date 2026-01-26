"""
RepRank: Representation rank diagnostics for actor/critic networks.

Implements effective rank and stable rank computation on network activations
to compare with the Rsubt certificate as baseline indicators.
"""

from __future__ import annotations

import torch
import numpy as np


def compute_effective_rank(
    activations: torch.Tensor,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute effective (numerical) rank of activation matrix.
    
    Effective rank is the number of singular values above a threshold,
    normalized by the largest singular value.
    
    Args:
        activations: Activation matrix of shape (batch, features).
        epsilon: Threshold for considering a singular value as non-zero.
    
    Returns:
        Effective rank as a float.
    """
    if activations.dim() != 2:
        activations = activations.view(activations.size(0), -1)
    
    # Center the activations
    activations = activations - activations.mean(dim=0, keepdim=True)
    
    # SVD
    try:
        s = torch.linalg.svdvals(activations.float())
    except Exception:
        return 0.0
    
    # Normalize by max singular value
    s_norm = s / (s[0] + epsilon)
    
    # Count singular values above threshold
    effective_rank = (s_norm > epsilon).sum().item()
    
    return effective_rank


def compute_stable_rank(
    activations: torch.Tensor,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute stable rank of activation matrix.
    
    Stable rank = ||A||_F^2 / ||A||_2^2 = sum(σ_i^2) / σ_1^2
    
    This is a continuous, differentiable proxy for rank that is more
    robust to noise than counting singular values.
    
    Args:
        activations: Activation matrix of shape (batch, features).
        epsilon: Numerical stability constant.
    
    Returns:
        Stable rank as a float.
    """
    if activations.dim() != 2:
        activations = activations.view(activations.size(0), -1)
    
    # Center the activations
    activations = activations - activations.mean(dim=0, keepdim=True)
    
    # Compute norms
    frobenius_sq = torch.sum(activations ** 2).item()
    
    try:
        s = torch.linalg.svdvals(activations.float())
        spectral_sq = (s[0] ** 2).item()
    except Exception:
        return 0.0
    
    if spectral_sq < epsilon:
        return 0.0
    
    stable_rank = frobenius_sq / (spectral_sq + epsilon)
    
    return stable_rank


def compute_entropy_rank(
    activations: torch.Tensor,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute entropy-based effective rank.
    
    Uses the entropy of the normalized singular value distribution:
    rank_eff = exp(H) where H = -sum(p_i * log(p_i)) and p_i = σ_i^2 / sum(σ_j^2)
    
    Args:
        activations: Activation matrix of shape (batch, features).
        epsilon: Numerical stability constant.
    
    Returns:
        Entropy-based effective rank.
    """
    if activations.dim() != 2:
        activations = activations.view(activations.size(0), -1)
    
    # Center the activations
    activations = activations - activations.mean(dim=0, keepdim=True)
    
    try:
        s = torch.linalg.svdvals(activations.float())
    except Exception:
        return 0.0
    
    # Compute probabilities from singular values squared
    s_sq = s ** 2
    total = s_sq.sum() + epsilon
    p = s_sq / total
    
    # Filter out zeros for log computation
    p = p[p > epsilon]
    
    if len(p) == 0:
        return 0.0
    
    # Entropy
    entropy = -torch.sum(p * torch.log(p)).item()
    
    # Effective rank
    return np.exp(entropy)


class RepRankTracker:
    """
    Tracker for representation rank diagnostics.
    
    Computes effective rank and stable rank on actor/critic activations
    at specified intervals.
    
    Args:
        compute_interval: Compute ranks every N calls to update().
    """
    
    def __init__(self, compute_interval: int = 10):
        self.compute_interval = compute_interval
        self._call_count = 0
        
        self.actor_effective_rank: float = 0.0
        self.actor_stable_rank: float = 0.0
        self.critic_effective_rank: float = 0.0
        self.critic_stable_rank: float = 0.0
    
    def update(
        self,
        actor_activations: torch.Tensor | None = None,
        critic_activations: torch.Tensor | None = None,
    ) -> dict | None:
        """
        Update rank diagnostics.
        
        Args:
            actor_activations: Actor hidden layer activations, shape (batch, features).
            critic_activations: Critic hidden layer activations, shape (batch, features).
        
        Returns:
            Dictionary with rank metrics if computed this call, else None.
        """
        self._call_count += 1
        
        if self._call_count % self.compute_interval != 0:
            return None
        
        results = {}
        
        if actor_activations is not None:
            self.actor_effective_rank = compute_effective_rank(actor_activations)
            self.actor_stable_rank = compute_stable_rank(actor_activations)
            results["actor_effrank"] = self.actor_effective_rank
            results["actor_stablerank"] = self.actor_stable_rank
        
        if critic_activations is not None:
            self.critic_effective_rank = compute_effective_rank(critic_activations)
            self.critic_stable_rank = compute_stable_rank(critic_activations)
            results["critic_effrank"] = self.critic_effective_rank
            results["critic_stablerank"] = self.critic_stable_rank
        
        return results if results else None
    
    def get_latest(self) -> dict:
        """Get most recent rank values."""
        return {
            "actor_effrank": self.actor_effective_rank,
            "actor_stablerank": self.actor_stable_rank,
            "critic_effrank": self.critic_effective_rank,
            "critic_stablerank": self.critic_stable_rank,
        }
