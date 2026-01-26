"""
EigenTracker: Eigendecomposition and subspace tracking for update covariance.

Builds the m×m Gram matrix from sketched updates, computes eigenvalues/vectors,
tracks top-k subspace basis, and computes the gap statistic for stability
certificate.
"""

from __future__ import annotations

import torch


class EigenTracker:
    """
    Tracks eigenspace of the update second-moment matrix in sketch space.
    
    Computes eigendecomposition of the m×m Gram matrix X @ X.T where X is
    the matrix of sketched updates, extracts top-k eigenvectors, and computes
    robust gap statistics.
    
    Args:
        k_max: Maximum number of eigenvalues/vectors to track.
        epsilon: Small constant for numerical stability in gap computation.
        device: PyTorch device for computations.
    """
    
    def __init__(
        self,
        k_max: int = 20,
        epsilon: float = 1e-8,
        device: str | torch.device = "cpu",
    ):
        self.k_max = k_max
        self.epsilon = epsilon
        self.device = torch.device(device)
        
        # Previous basis for shock computation
        self.prev_basis: torch.Tensor | None = None
        self.prev_projector: torch.Tensor | None = None
        
        # Current eigenvalues and basis
        self.eigenvalues: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None  # Basis in sketch space
        self.projector: torch.Tensor | None = None  # Pi = U @ U.T
    
    def update(self, sketches: torch.Tensor) -> dict:
        """
        Update eigendecomposition from buffer of sketches.
        
        Computes the Gram matrix, eigendecomposition, extracts top-k basis,
        and computes gap statistics.
        
        Args:
            sketches: Matrix of sketched updates, shape (m, sketch_dim).
        
        Returns:
            Dictionary with eigenvalues, gaps, and gap_ratio.
        """
        m, s = sketches.shape
        sketches = sketches.to(dtype=torch.float32, device=self.device)
        
        # Build Gram matrix: G = (1/m) * X @ X.T, shape (m, m)
        # We'll work with the m×m matrix for efficiency when m < s
        gram = (sketches @ sketches.T) / m
        
        # Eigendecomposition (symmetric, so use eigh)
        # Returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        
        # Reverse to descending order
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        
        # Keep top k_max
        k = min(self.k_max, len(eigenvalues) - 1)
        top_eigenvalues = eigenvalues[:k + 1]  # Need k+1 for gap computation
        top_eigenvectors = eigenvectors[:, :k]  # m × k
        
        # Convert eigenvectors of Gram to eigenvectors in sketch space
        # If G = X @ X.T has eigenvector v with eigenvalue λ,
        # then X.T @ v / sqrt(λ) is eigenvector of X.T @ X with same λ
        # Basis in sketch space: U = X.T @ V @ Λ^{-1/2}
        valid_mask = top_eigenvalues[:k] > self.epsilon
        if valid_mask.sum() == 0:
            # All eigenvalues too small, return early
            self.eigenvalues = eigenvalues
            return {
                "eigenvalues": eigenvalues.cpu().numpy(),
                "gaps": [],
                "gap_ratio": 0.0,
                "min_gap": 0.0,
            }
        
        # Compute basis in sketch space for valid eigenvalues
        valid_k = valid_mask.sum().item()
        inv_sqrt_lambda = 1.0 / torch.sqrt(top_eigenvalues[:valid_k] + self.epsilon)
        basis_sketch = (sketches.T @ top_eigenvectors[:, :valid_k]) * inv_sqrt_lambda
        # Normalize columns
        basis_sketch = basis_sketch / (torch.norm(basis_sketch, dim=0, keepdim=True) + self.epsilon)
        
        # Save previous for shock computation
        self.prev_basis = self.basis
        self.prev_projector = self.projector
        
        # Update current
        self.eigenvalues = eigenvalues
        self.basis = basis_sketch  # sketch_dim × valid_k
        self.projector = basis_sketch @ basis_sketch.T  # sketch_dim × sketch_dim
        
        # Compute gap statistics
        gaps = []
        for i in range(min(k, len(eigenvalues) - 1)):
            gap = (eigenvalues[i] - eigenvalues[i + 1]).item()
            gaps.append(gap)
        
        # Robust min-gap statistic
        if gaps:
            min_gap = min(gaps)
            lambda1 = eigenvalues[0].item()
            gap_ratio = min_gap / (lambda1 + self.epsilon)
        else:
            min_gap = 0.0
            gap_ratio = 0.0
        
        return {
            "eigenvalues": eigenvalues.cpu().numpy(),
            "gaps": gaps,
            "gap_ratio": gap_ratio,
            "min_gap": min_gap,
        }
    
    def compute_shock(self, sketch: torch.Tensor) -> float:
        """
        Compute directional shock α_t(v) = ||(Π_t - Π_{t-1}) v|| / ||v||.
        
        This is the Lemma 3 quantity: the directional shock seen by the actual update.
        
        Args:
            sketch: Current sketched update vector, shape (sketch_dim,).
        
        Returns:
            Directional shock value in [0, 1].
        """
        if self.projector is None or self.prev_projector is None:
            return 0.0
        
        sketch = sketch.to(dtype=torch.float32, device=self.device)
        norm = torch.norm(sketch)
        
        if norm < 1e-12:
            return 0.0
        
        # Compute (Π_t - Π_{t-1}) @ sketch
        diff_proj = self.projector - self.prev_projector
        projected = diff_proj @ sketch
        
        shock = torch.norm(projected) / norm
        return shock.item()
    
    def has_previous(self) -> bool:
        """Check if we have a previous projector for shock computation."""
        return self.prev_projector is not None
