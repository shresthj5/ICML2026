"""
Sketcher: Random coordinate subset sketching for high-dimensional parameter vectors.

Implements random index selection + gather for projecting d-dimensional parameter
updates into a lower s-dimensional sketch space, with proper scaling by sqrt(d/s).
"""

from __future__ import annotations

import torch
import numpy as np


class Sketcher:
    """
    Random coordinate sketcher for parameter vectors.
    
    Projects d-dimensional vectors into s-dimensional sketch space using
    a fixed random subset of coordinates, scaled by sqrt(d/s) for unbiased
    norm preservation in expectation.
    
    Args:
        param_dim: Total number of parameters (d).
        sketch_dim: Dimension of sketch space (s). Will be clamped to param_dim.
        device: PyTorch device for sketch operations.
        seed: Random seed for reproducible index selection.
    """
    
    def __init__(
        self,
        param_dim: int,
        sketch_dim: int,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ):
        self.param_dim = param_dim
        self.sketch_dim = min(sketch_dim, param_dim)
        self.device = torch.device(device)
        self.seed = seed
        
        # Generate fixed random indices
        rng = np.random.default_rng(seed)
        indices = rng.choice(param_dim, size=self.sketch_dim, replace=False)
        indices = np.sort(indices)  # sorted for potentially better cache locality
        self.indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        # Scaling factor for unbiased norm preservation
        self.scale = np.sqrt(param_dim / self.sketch_dim)
    
    def sketch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a parameter vector into sketch space.
        
        Args:
            x: Flat parameter vector of shape (param_dim,) or (batch, param_dim).
        
        Returns:
            Sketched vector of shape (sketch_dim,) or (batch, sketch_dim).
        """
        if x.dim() == 1:
            return x[self.indices] * self.scale
        else:
            return x[:, self.indices] * self.scale
    
    def sketch_from_params(self, params: list[torch.nn.Parameter]) -> torch.Tensor:
        """
        Extract and sketch parameters from a list of nn.Parameters.
        
        Args:
            params: List of parameter tensors (will be flattened and concatenated).
        
        Returns:
            Sketched vector of shape (sketch_dim,).
        """
        flat = torch.cat([p.data.view(-1) for p in params])
        return self.sketch(flat)
    
    def get_param_vector(self, params: list[torch.nn.Parameter]) -> torch.Tensor:
        """
        Flatten and concatenate parameters into a single vector.
        
        Args:
            params: List of parameter tensors.
        
        Returns:
            Flat vector of shape (param_dim,).
        """
        return torch.cat([p.data.view(-1) for p in params])


def count_parameters(params: list[torch.nn.Parameter]) -> int:
    """Count total number of parameters."""
    return sum(p.numel() for p in params)
