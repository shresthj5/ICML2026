"""
RsubtMonitor: Main interface combining all Rsubt components.

Provides a simple API for integration into training scripts:
- update(): Feed new parameter updates
- maybe_compute(): Compute certificate metrics when ready
- alarm_state(): Get current alarm state
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
from pathlib import Path

import torch

from cleanrl_utils.rsubt.sketch import Sketcher, count_parameters
from cleanrl_utils.rsubt.buffer import UpdateBuffer
from cleanrl_utils.rsubt.tracker import EigenTracker
from cleanrl_utils.rsubt.certificate import RsubtCertificate, AlarmState
from cleanrl_utils.rsubt.controller import RiskController, PPOHyperparams, SACHyperparams, TD3Hyperparams

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class RsubtMonitor:
    """
    Main Rsubt monitoring interface.
    
    Combines sketcher, buffer, eigentracker, certificate, and controller
    into a unified API for easy integration with training scripts.
    
    Args:
        params: List of actor parameters to monitor.
        sketch_dim: Dimension of sketch space (default 8192 for MLP, 32768 for CNN).
        buffer_size: Number of updates to store (m).
        k_max: Maximum eigenvalues to track.
        diag_interval: DEPRECATED - use diag_interval_steps instead.
        diag_interval_steps: Compute diagnostics every N environment steps (default 20480).
        min_buffer_entries: Minimum buffer entries before computing (default 4).
        algorithm: "ppo", "sac", or "td3" for controller.
        device: PyTorch device.
        seed: Random seed for sketcher.
        enable_controller: Whether to enable risk-gated control.
        **controller_kwargs: Additional kwargs for RiskController.
    """
    
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        sketch_dim: int = 8192,
        buffer_size: int = 64,
        k_max: int = 20,
        diag_interval: int = 10,  # DEPRECATED, kept for backward compat
        diag_interval_steps: int = 20480,  # NEW: step-based interval
        min_buffer_entries: int = 4,  # NEW: minimum buffer entries for warmup
        algorithm: str = "ppo",
        device: str | torch.device = "cpu",
        seed: int = 42,
        enable_controller: bool = True,
        tau_yellow: float | None = None,
        tau_red: float | None = None,
        thresholds_path: str | Path | None = None,
        **controller_kwargs,
    ):
        self.device = torch.device(device)
        self.diag_interval = diag_interval  # kept for backward compat
        self.diag_interval_steps = diag_interval_steps
        self.min_buffer_entries = min_buffer_entries
        self.enable_controller = enable_controller
        
        # Count parameters and initialize sketcher
        param_dim = count_parameters(params)
        self.sketcher = Sketcher(param_dim, sketch_dim, device, seed)
        
        # Initialize buffer
        self.buffer = UpdateBuffer(
            capacity=buffer_size,
            sketch_dim=self.sketcher.sketch_dim,
            device=device,
            dtype=torch.float16,  # Save memory
        )
        
        # Initialize tracker
        self.tracker = EigenTracker(k_max=k_max, device=device)
        
        # Initialize certificate
        self.certificate = RsubtCertificate()
        if thresholds_path is not None and str(thresholds_path):
            self.load_thresholds(thresholds_path)
        if (tau_yellow is None) ^ (tau_red is None):
            raise ValueError("tau_yellow and tau_red must be set together, or both left as None")
        if tau_yellow is not None and tau_red is not None:
            self.set_thresholds(tau_yellow=tau_yellow, tau_red=tau_red)
        
        # Initialize controller
        if enable_controller:
            self.controller = RiskController(algorithm=algorithm, **controller_kwargs)
        else:
            self.controller = None
        
        # State tracking
        self._update_count = 0
        self._prev_params: torch.Tensor | None = None
        self._last_metrics: dict | None = None
        self._last_log_step: int = 0  # Track last step we logged at
    
    def snapshot_params(self, params: list[torch.nn.Parameter]) -> None:
        """
        Snapshot current parameters before an update.
        
        Call this before the optimizer step(s) to capture θ_t.
        
        Args:
            params: List of actor parameters.
        """
        self._prev_params = self.sketcher.get_param_vector(params).detach().clone()
    
    def update(
        self,
        params: list[torch.nn.Parameter],
        global_step: int,
    ) -> None:
        """
        Record a parameter update.
        
        Call this after the optimizer step(s) to compute Δθ = θ_{t+1} - θ_t.
        
        Args:
            params: List of actor parameters (after update).
            global_step: Current global step for logging.
        """
        if self._prev_params is None:
            # First call, just snapshot
            self.snapshot_params(params)
            return
        
        # Compute update
        current_params = self.sketcher.get_param_vector(params)
        delta = current_params - self._prev_params
        
        # Compute norm before sketching
        update_norm = torch.norm(delta).item()
        
        # Sketch the update
        sketch = self.sketcher.sketch(delta)
        
        # Add to buffer
        self.buffer.add(sketch, update_norm, global_step)
        
        self._update_count += 1
        
        # Snapshot for next update
        self._prev_params = current_params.detach().clone()
    
    def maybe_compute(self, global_step: int = None) -> dict | None:
        """
        Compute certificate metrics if it's time.
        
        Uses step-based intervals for consistent behavior regardless of num_envs.
        
        Args:
            global_step: Current environment step count. If None, falls back to
                         iteration-based logic (deprecated).
        
        Returns:
            Dictionary with metrics if computed, else None.
        """
        # Step-based logic (preferred)
        if global_step is not None:
            # Need minimum buffer entries for eigenvalue computation
            if self.buffer.size() < self.min_buffer_entries:
                return None
            
            # Check if enough steps have passed since last log
            if global_step - self._last_log_step < self.diag_interval_steps:
                return None
        else:
            # Legacy iteration-based logic (deprecated, for backward compat)
            if self._update_count % self.diag_interval != 0:
                return None
            
            if not self.buffer.is_full():
                return None
        
        # Get all sketches
        sketches, norms, indices = self.buffer.get_all()
        
        # Update eigentracker
        eigen_info = self.tracker.update(sketches)
        
        # Get latest sketch for shock computation
        latest = self.buffer.get_latest()
        if latest is None:
            return None
        
        latest_sketch, latest_norm, latest_idx = latest
        
        # Compute shock
        shock = self.tracker.compute_shock(latest_sketch)
        
        # Update certificate
        cert_info = self.certificate.update(shock, eigen_info["gap_ratio"])
        
        # Combine metrics
        metrics = {
            "raw": cert_info["raw"],
            "ewma": cert_info["ewma"],
            "state": cert_info["state"],
            "state_int": cert_info["state_int"],
            "shock": shock,
            "gap_ratio": eigen_info["gap_ratio"],
            "min_gap": eigen_info["min_gap"],
            "update_norm": latest_norm,
            "lambda_1": eigen_info["eigenvalues"][0] if len(eigen_info["eigenvalues"]) > 0 else 0.0,
        }
        
        self._last_metrics = metrics
        # Update last log step for step-based interval tracking
        if global_step is not None:
            self._last_log_step = global_step
        return metrics
    
    def alarm_state(self) -> AlarmState:
        """Get current alarm state."""
        return self.certificate.state

    def set_thresholds(self, tau_yellow: float, tau_red: float) -> None:
        """Set certificate thresholds (overrides defaults)."""
        self.certificate.set_thresholds(tau_yellow=tau_yellow, tau_red=tau_red)

    def load_thresholds(self, thresholds_path: str | Path, metric_key: str = "rsubt/ewma") -> None:
        """
        Load thresholds from a JSON file produced by `cleanrl_utils.rsubt.calibrate_thresholds`.
        
        Supports either:
        - Top-level keys: {"tau_yellow": ..., "tau_red": ...}
        - Per-metric keys: {"metrics": {"rsubt/ewma": {"tau_yellow": ..., "tau_red": ...}}}
        """
        path = Path(thresholds_path)
        with path.open("r") as f:
            data = json.load(f)
        tau_yellow = data.get("tau_yellow")
        tau_red = data.get("tau_red")
        if tau_yellow is None or tau_red is None:
            metric = (data.get("metrics") or {}).get(metric_key) or {}
            tau_yellow = metric.get("tau_yellow", tau_yellow)
            tau_red = metric.get("tau_red", tau_red)
        if tau_yellow is None or tau_red is None:
            raise ValueError(
                f"Thresholds file {path} missing tau_yellow/tau_red "
                f"(looked for top-level keys and metrics['{metric_key}'])"
            )
        self.set_thresholds(tau_yellow=float(tau_yellow), tau_red=float(tau_red))
    
    def get_hyperparams(self) -> PPOHyperparams | SACHyperparams | TD3Hyperparams | None:
        """
        Get risk-adjusted hyperparameters.
        
        Returns:
            Adjusted hyperparameters based on current alarm state,
            or None if controller is disabled.
        """
        if self.controller is None:
            return None
        return self.controller.get_hyperparams(self.certificate.state)
    
    def get_last_metrics(self) -> dict | None:
        """Get the most recently computed metrics."""
        return self._last_metrics
    
    def log_to_tensorboard(
        self,
        writer: "SummaryWriter",
        global_step: int,
    ) -> None:
        """
        Log current metrics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter.
            global_step: Current global step.
        """
        from cleanrl_utils.rsubt.io import log_rsubt_metrics
        
        if self._last_metrics is None:
            return
        
        log_rsubt_metrics(
            writer=writer,
            global_step=global_step,
            raw=self._last_metrics["raw"],
            ewma=self._last_metrics["ewma"],
            state=self._last_metrics["state"],
            shock=self._last_metrics["shock"],
            gap_ratio=self._last_metrics["gap_ratio"],
            update_norm=self._last_metrics["update_norm"],
        )
    
    def reset(self) -> None:
        """Reset all state."""
        self.buffer.clear()
        self.certificate.reset()
        if self.controller:
            self.controller.reset()
        self._update_count = 0
        self._prev_params = None
        self._last_metrics = None
        self._last_log_step = 0
