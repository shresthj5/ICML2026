"""
UpdateBuffer: Ring buffer for storing recent parameter update sketches.

Stores the last m sketches of actor parameter updates along with their norms
and update indices, enabling computation of the sliding-window second-moment
matrix for eigengap tracking.
"""

from __future__ import annotations

import torch


class UpdateBuffer:
    """
    Ring buffer for recent update sketches.
    
    Stores the last m update sketches in a fixed-size tensor buffer,
    along with original (pre-sketch) norms and update indices.
    
    Args:
        capacity: Number of updates to store (m).
        sketch_dim: Dimension of sketch space.
        device: PyTorch device for storage.
        dtype: Data type for storage (default float32; use float16 for memory).
    """
    
    def __init__(
        self,
        capacity: int,
        sketch_dim: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = capacity
        self.sketch_dim = sketch_dim
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Pre-allocate storage
        self.sketches = torch.zeros(
            (capacity, sketch_dim), dtype=dtype, device=self.device
        )
        self.norms = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.indices = torch.zeros(capacity, dtype=torch.long, device=self.device)
        
        self.pos = 0  # Current write position
        self.count = 0  # Number of items stored (up to capacity)
    
    def add(
        self,
        sketch: torch.Tensor,
        norm: float,
        update_idx: int,
    ) -> None:
        """
        Add a new update sketch to the buffer.
        
        Args:
            sketch: Sketched update vector of shape (sketch_dim,).
            norm: L2 norm of the original (pre-sketch) update.
            update_idx: Global update index / step number.
        """
        self.sketches[self.pos] = sketch.to(dtype=self.dtype, device=self.device)
        self.norms[self.pos] = norm
        self.indices[self.pos] = update_idx
        
        self.pos = (self.pos + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    
    def get_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all stored sketches in chronological order.
        
        Returns:
            Tuple of (sketches, norms, indices), each of shape (count, ...).
        """
        if self.count < self.capacity:
            return (
                self.sketches[:self.count],
                self.norms[:self.count],
                self.indices[:self.count],
            )
        else:
            # Reorder to chronological: oldest first
            order = torch.cat([
                torch.arange(self.pos, self.capacity, device=self.device),
                torch.arange(0, self.pos, device=self.device),
            ])
            return (
                self.sketches[order],
                self.norms[order],
                self.indices[order],
            )
    
    def get_latest(self) -> tuple[torch.Tensor, float, int] | None:
        """
        Get the most recently added sketch.
        
        Returns:
            Tuple of (sketch, norm, index) or None if buffer is empty.
        """
        if self.count == 0:
            return None
        idx = (self.pos - 1) % self.capacity
        return (
            self.sketches[idx],
            self.norms[idx].item(),
            self.indices[idx].item(),
        )
    
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return self.count >= self.capacity
    
    def size(self) -> int:
        """Get current number of stored items."""
        return self.count
    
    def clear(self) -> None:
        """Reset buffer to empty state."""
        self.pos = 0
        self.count = 0
        self.sketches.zero_()
        self.norms.zero_()
        self.indices.zero_()
