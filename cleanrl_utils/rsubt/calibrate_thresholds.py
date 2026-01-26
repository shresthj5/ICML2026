"""
Calibrate Rsubt alarm thresholds from TensorBoard runs.

Reads TensorBoard event files from a calibration set of runs and
determines threshold values τ_Y and τ_R that achieve a target
false positive rate (FPR) on non-collapse runs.

Usage:
    python -m cleanrl_utils.rsubt.calibrate_thresholds \
        --runs_dir runs/calibration/ \
        --target_fpr 0.05 \
        --output thresholds.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise ImportError("tensorboard is required. Install with: pip install tensorboard")


def load_scalar_from_events(
    logdir: str | Path,
    scalar_key: str,
) -> tuple[list[int], list[float]]:
    """
    Load a scalar time series from TensorBoard event files.
    
    Args:
        logdir: Directory containing event files.
        scalar_key: Key of the scalar to load (e.g., "rsubt/ewma").
    
    Returns:
        Tuple of (steps, values) lists.
    """
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    
    if scalar_key not in ea.Tags()["scalars"]:
        return [], []
    
    events = ea.Scalars(scalar_key)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return steps, values


def detect_collapse(
    steps: list[int],
    returns: list[float],
    threshold_ratio: float = 0.2,
    consecutive_evals: int = 3,
) -> int | None:
    """
    Detect collapse time from return history.
    
    Collapse is defined as:
    - Mean return < threshold_ratio * max_return for consecutive_evals evals.
    
    Args:
        steps: Global step for each evaluation.
        returns: Episodic return at each evaluation.
        threshold_ratio: Fraction of peak return below which is "collapsed".
        consecutive_evals: Number of consecutive low evals needed.
    
    Returns:
        Collapse step, or None if no collapse detected.
    """
    if len(returns) < consecutive_evals:
        return None
    
    max_return = float("-inf")
    low_count = 0
    
    for i, (step, ret) in enumerate(zip(steps, returns)):
        # Update running max
        if ret > max_return:
            max_return = ret
            low_count = 0
        else:
            # Check if below threshold
            threshold = threshold_ratio * max_return
            if ret < threshold:
                low_count += 1
                if low_count >= consecutive_evals:
                    # Collapse detected at the first of the consecutive low evals
                    collapse_idx = i - consecutive_evals + 1
                    return steps[collapse_idx]
            else:
                low_count = 0
    
    return None


def compute_fpr_threshold(
    all_values: list[list[float]],
    target_fpr: float,
) -> float:
    """
    Compute threshold that achieves target FPR.
    
    Args:
        all_values: List of value sequences from non-collapse runs.
        target_fpr: Target false positive rate (e.g., 0.05).
    
    Returns:
        Threshold value.
    """
    # Flatten all values
    flat_values = []
    for values in all_values:
        flat_values.extend(values)
    
    if not flat_values:
        return 1.0
    
    # Threshold at (1 - target_fpr) percentile
    threshold = np.percentile(flat_values, 100 * (1 - target_fpr))
    
    return float(threshold)


def calibrate_from_runs(
    runs_dir: str | Path,
    rsubt_key: str = "rsubt/ewma",
    return_key: str = "charts/episodic_return",
    target_fpr: float = 0.05,
    collapse_threshold_ratio: float = 0.2,
    yellow_red_ratio: float = 2.0,
) -> dict[str, float]:
    """
    Calibrate thresholds from a directory of TensorBoard runs.
    
    Args:
        runs_dir: Directory containing run subdirectories.
        rsubt_key: TensorBoard key for Rsubt EWMA values.
        return_key: TensorBoard key for episodic returns.
        target_fpr: Target false positive rate.
        collapse_threshold_ratio: Ratio for collapse detection.
        yellow_red_ratio: Ratio between red and yellow thresholds.
    
    Returns:
        Dictionary with tau_yellow and tau_red values.
    """
    runs_dir = Path(runs_dir)
    
    non_collapse_rsubt_values = []
    collapse_runs = []
    non_collapse_runs = []
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Load return data
        return_steps, return_values = load_scalar_from_events(run_dir, return_key)
        if not return_values:
            continue
        
        # Load Rsubt data
        rsubt_steps, rsubt_values = load_scalar_from_events(run_dir, rsubt_key)
        if not rsubt_values:
            continue
        
        # Detect collapse
        collapse_step = detect_collapse(
            return_steps, return_values,
            threshold_ratio=collapse_threshold_ratio,
        )
        
        if collapse_step is None:
            # Non-collapse run: use for threshold calibration
            non_collapse_runs.append(run_dir.name)
            non_collapse_rsubt_values.append(rsubt_values)
        else:
            collapse_runs.append(run_dir.name)
    
    print(f"Found {len(non_collapse_runs)} non-collapse runs for calibration")
    print(f"Found {len(collapse_runs)} collapse runs")
    
    if not non_collapse_rsubt_values:
        print("Warning: No non-collapse runs found. Using default thresholds.")
        return {"tau_yellow": 0.5, "tau_red": 1.0}
    
    # Compute yellow threshold at target FPR
    tau_yellow = compute_fpr_threshold(non_collapse_rsubt_values, target_fpr)
    
    # Red threshold is a multiple of yellow
    tau_red = tau_yellow * yellow_red_ratio
    
    return {
        "tau_yellow": tau_yellow,
        "tau_red": tau_red,
        "target_fpr": target_fpr,
        "n_non_collapse_runs": len(non_collapse_runs),
        "n_collapse_runs": len(collapse_runs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Rsubt alarm thresholds from TensorBoard runs"
    )
    parser.add_argument(
        "--runs_dir", type=str, required=True,
        help="Directory containing TensorBoard run subdirectories"
    )
    parser.add_argument(
        "--target_fpr", type=float, default=0.05,
        help="Target false positive rate (default: 0.05)"
    )
    parser.add_argument(
        "--rsubt_key", type=str, default="rsubt/ewma",
        help="TensorBoard key for Rsubt EWMA values"
    )
    parser.add_argument(
        "--return_key", type=str, default="charts/episodic_return",
        help="TensorBoard key for episodic returns"
    )
    parser.add_argument(
        "--output", type=str, default="thresholds.json",
        help="Output JSON file for thresholds"
    )
    parser.add_argument(
        "--yellow_red_ratio", type=float, default=2.0,
        help="Ratio between red and yellow thresholds"
    )
    
    args = parser.parse_args()
    
    thresholds = calibrate_from_runs(
        runs_dir=args.runs_dir,
        rsubt_key=args.rsubt_key,
        return_key=args.return_key,
        target_fpr=args.target_fpr,
        yellow_red_ratio=args.yellow_red_ratio,
    )
    
    print(f"\nCalibrated thresholds:")
    print(f"  tau_yellow: {thresholds['tau_yellow']:.4f}")
    print(f"  tau_red: {thresholds['tau_red']:.4f}")
    
    with open(args.output, "w") as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
