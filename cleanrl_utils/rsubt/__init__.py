"""
Rsubt: Eigengap collapse and subspace rotation certificate for policy gradient instability.

This module implements the stability certificate based on eigengap collapse detection
and directional shock measurement, as described in the paper "Eigengap Collapse and
Subspace Rotation as a Mechanism for Policy Gradient Instability".

Core components:
- Sketcher: Random coordinate sketching for high-dimensional parameters
- UpdateBuffer: Ring buffer for storing recent update sketches
- EigenTracker: Eigendecomposition and subspace tracking
- RsubtCertificate: Stability certificate with EWMA and hysteresis
- RiskController: Risk-gated hyperparameter controller
- RsubtMonitor: Main monitoring interface combining all components

Evaluation tools:
- calibrate_thresholds: Calibrate alarm thresholds from TensorBoard runs
- evaluate_early_warning: Evaluate prediction performance (lead time, TPR, AUC)
- benchmarks: Generate benchmark command recipes

Diagnostics:
- reprank: Representation rank diagnostics (effective rank, stable rank)
"""

from cleanrl_utils.rsubt.sketch import Sketcher
from cleanrl_utils.rsubt.buffer import UpdateBuffer
from cleanrl_utils.rsubt.tracker import EigenTracker
from cleanrl_utils.rsubt.certificate import RsubtCertificate, AlarmState
from cleanrl_utils.rsubt.controller import RiskController, PPOHyperparams, SACHyperparams
from cleanrl_utils.rsubt.monitor import RsubtMonitor
from cleanrl_utils.rsubt.reprank import (
    compute_effective_rank,
    compute_stable_rank,
    compute_entropy_rank,
    RepRankTracker,
)

__all__ = [
    # Core components
    "Sketcher",
    "UpdateBuffer",
    "EigenTracker",
    "RsubtCertificate",
    "AlarmState",
    "RiskController",
    "PPOHyperparams",
    "SACHyperparams",
    "RsubtMonitor",
    # Rank diagnostics
    "compute_effective_rank",
    "compute_stable_rank",
    "compute_entropy_rank",
    "RepRankTracker",
]
