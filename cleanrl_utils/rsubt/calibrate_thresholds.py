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

Supports calibrating multiple metrics, not just Rsubt.
"""

from __future__ import annotations

import argparse
import json
import re
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
    
    tags = ea.Tags().get("scalars", [])
    if scalar_key not in tags:
        return [], []
    
    events = ea.Scalars(scalar_key)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return steps, values


def compute_ema(values: list[float], alpha: float = 0.2) -> list[float]:
    """
    Compute exponential moving average of values.
    
    Args:
        values: Input values.
        alpha: Smoothing factor (higher = more weight on recent).
    
    Returns:
        EMA-smoothed values.
    """
    if not values:
        return []
    
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def detect_collapse_ema(
    steps: list[int],
    returns: list[float],
    ema_window: int = 5,
    drop_ratio: float = 0.4,
    persistence: int = 3,
) -> int | None:
    """
    Detect collapse time from return history using EMA-based definition.
    
    Collapse is defined as:
    - EMA(R(t)) <= (1 - drop_ratio) * R_best(t) for `persistence` consecutive evals.
    - R_best(t) = max_{s<=t} EMA(R(s))
    
    This handles both positive and negative return tasks by using relative drops.
    
    Args:
        steps: Global step for each evaluation.
        returns: Return at each evaluation (eval/return_mean).
        ema_window: Window for EMA (controls alpha = 2/(w+1)).
        drop_ratio: Fraction of peak to consider as collapse (e.g., 0.4 = 40% drop).
        persistence: Number of consecutive eval points below threshold.
    
    Returns:
        Collapse step, or None if no collapse detected.
    """
    if len(returns) < persistence:
        return None
    
    # Compute EMA
    alpha = 2.0 / (ema_window + 1)
    ema = compute_ema(returns, alpha)
    
    # Track running best and consecutive low counts
    r_best = ema[0]
    low_count = 0
    
    for i, (step, ema_val) in enumerate(zip(steps, ema)):
        # Update running best
        if ema_val > r_best:
            r_best = ema_val
            low_count = 0
        else:
            # Compute threshold based on sign of r_best
            # For positive r_best: threshold = (1 - drop_ratio) * r_best
            # For negative r_best: threshold = (1 + drop_ratio) * r_best (more negative)
            if r_best > 0:
                threshold = (1 - drop_ratio) * r_best
                is_collapsed = ema_val < threshold
            elif r_best < 0:
                # For negative rewards, "collapse" means getting even more negative
                threshold = (1 + drop_ratio) * r_best
                is_collapsed = ema_val < threshold
            else:
                # r_best == 0: any significant negative is collapse
                threshold = -abs(drop_ratio)
                is_collapsed = ema_val < threshold
            
            if is_collapsed:
                low_count += 1
                if low_count >= persistence:
                    # Collapse detected at the first of the consecutive low evals
                    collapse_idx = i - persistence + 1
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


def parse_run_name(run_name: str) -> dict[str, Any]:
    """
    Parse run name to extract env, algorithm, seed, timestamp.
    
    Expected format: EnvId__script_name__seed__timestamp
    """
    parts = run_name.split("__")
    result = {"env": None, "algo": None, "seed": None, "timestamp": None}
    
    if len(parts) >= 1:
        result["env"] = parts[0]
    if len(parts) >= 2:
        result["algo"] = parts[1]
    if len(parts) >= 3:
        try:
            result["seed"] = int(parts[2])
        except ValueError:
            result["seed"] = parts[2]
    if len(parts) >= 4:
        result["timestamp"] = parts[3]
    
    return result


def split_runs_dev_test(
    run_dirs: list[Path],
    dev_envs: list[str] | None = None,
    dev_seeds: list[int] | None = None,
    dev_ratio: float = 0.5,
) -> tuple[list[Path], list[Path]]:
    """
    Split runs into dev (calibration) and test (held-out) sets.
    
    Args:
        run_dirs: List of run directories.
        dev_envs: Specific envs for dev set (if None, split by ratio).
        dev_seeds: Specific seeds for dev set (if None, split by ratio).
        dev_ratio: Fraction of runs for dev set if not using explicit splits.
    
    Returns:
        Tuple of (dev_runs, test_runs).
    """
    dev_runs = []
    test_runs = []
    
    for run_dir in run_dirs:
        info = parse_run_name(run_dir.name)
        
        # Check explicit splits
        if dev_envs is not None and info["env"] in dev_envs:
            dev_runs.append(run_dir)
        elif dev_seeds is not None and info["seed"] in dev_seeds:
            dev_runs.append(run_dir)
        elif dev_envs is not None or dev_seeds is not None:
            test_runs.append(run_dir)
        else:
            # Random split based on seed hash
            if info["seed"] is not None:
                seed_hash = hash(str(info["seed"])) % 100
                if seed_hash < dev_ratio * 100:
                    dev_runs.append(run_dir)
                else:
                    test_runs.append(run_dir)
            else:
                # No seed info, alternate
                if len(dev_runs) <= len(test_runs):
                    dev_runs.append(run_dir)
                else:
                    test_runs.append(run_dir)
    
    return dev_runs, test_runs


def calibrate_from_runs(
    runs_dir: str | Path,
    return_key: str = "eval/return_mean",
    target_fpr: float = 0.05,
    ema_window: int = 5,
    drop_ratio: float = 0.4,
    persistence: int = 3,
    yellow_red_ratio: float = 2.0,
    metrics_to_calibrate: list[str] | None = None,
    dev_envs: list[str] | None = None,
    dev_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """
    Calibrate thresholds from a directory of TensorBoard runs.
    
    Args:
        runs_dir: Directory containing run subdirectories.
        return_key: TensorBoard key for returns (eval/return_mean preferred).
        target_fpr: Target false positive rate.
        ema_window: EMA window for collapse detection.
        drop_ratio: Drop ratio for collapse definition.
        persistence: Persistence for collapse definition.
        yellow_red_ratio: Ratio between red and yellow thresholds.
        metrics_to_calibrate: List of metrics to calibrate (default: rsubt/ewma + baselines).
        dev_envs: Specific envs for dev set.
        dev_seeds: Specific seeds for dev set.
    
    Returns:
        Dictionary with calibrated thresholds for all metrics.
    """
    runs_dir = Path(runs_dir)
    
    # Default metrics to calibrate
    if metrics_to_calibrate is None:
        metrics_to_calibrate = [
            "rsubt/ewma",
            "rsubt/shock",
            "losses/approx_kl",
            "losses/clipfrac",
            "losses/value_loss",
            "baseline/grad_norm_actor",
            "losses/entropy",
            "reprank/actor_stablerank",
        ]
    
    # Get all run directories
    all_run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    # Split into dev and test
    dev_runs, test_runs = split_runs_dev_test(all_run_dirs, dev_envs, dev_seeds)
    
    print(f"Total runs: {len(all_run_dirs)}")
    print(f"Dev runs: {len(dev_runs)}")
    print(f"Test runs: {len(test_runs)}")
    
    # Use dev runs for calibration
    non_collapse_metric_values = {m: [] for m in metrics_to_calibrate}
    collapse_runs = []
    non_collapse_runs = []
    
    for run_dir in dev_runs:
        # Load return data
        return_steps, return_values = load_scalar_from_events(run_dir, return_key)
        if not return_values:
            # Fallback to training returns
            return_steps, return_values = load_scalar_from_events(run_dir, "charts/episodic_return")
        if not return_values:
            continue
        
        # Detect collapse
        collapse_step = detect_collapse_ema(
            return_steps, return_values,
            ema_window=ema_window,
            drop_ratio=drop_ratio,
            persistence=persistence,
        )
        
        if collapse_step is None:
            # Non-collapse run: use for threshold calibration
            non_collapse_runs.append(run_dir.name)
            
            # Load all metric values
            for metric in metrics_to_calibrate:
                metric_steps, metric_values = load_scalar_from_events(run_dir, metric)
                if metric_values:
                    non_collapse_metric_values[metric].append(metric_values)
        else:
            collapse_runs.append(run_dir.name)
    
    print(f"\nDev set analysis:")
    print(f"  Non-collapse runs for calibration: {len(non_collapse_runs)}")
    print(f"  Collapse runs: {len(collapse_runs)}")
    
    # Compute thresholds for each metric
    thresholds = {
        "target_fpr": target_fpr,
        "n_dev_runs": len(dev_runs),
        "n_test_runs": len(test_runs),
        "n_non_collapse_runs": len(non_collapse_runs),
        "n_collapse_runs": len(collapse_runs),
        "collapse_config": {
            "ema_window": ema_window,
            "drop_ratio": drop_ratio,
            "persistence": persistence,
        },
        "metrics": {},
    }
    
    for metric in metrics_to_calibrate:
        values = non_collapse_metric_values[metric]
        if values:
            tau = compute_fpr_threshold(values, target_fpr)
            thresholds["metrics"][metric] = {
                "tau_yellow": tau,
                "tau_red": tau * yellow_red_ratio,
                "n_runs_with_metric": len(values),
            }
            print(f"  {metric}: tau_yellow={tau:.4f}, tau_red={tau * yellow_red_ratio:.4f}")
        else:
            print(f"  {metric}: No data available")
    
    # Legacy keys for backward compatibility
    if "rsubt/ewma" in thresholds["metrics"]:
        thresholds["tau_yellow"] = thresholds["metrics"]["rsubt/ewma"]["tau_yellow"]
        thresholds["tau_red"] = thresholds["metrics"]["rsubt/ewma"]["tau_red"]
    
    return thresholds


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
        "--return_key", type=str, default="eval/return_mean",
        help="TensorBoard key for returns (default: eval/return_mean)"
    )
    parser.add_argument(
        "--output", type=str, default="thresholds.json",
        help="Output JSON file for thresholds"
    )
    parser.add_argument(
        "--yellow_red_ratio", type=float, default=2.0,
        help="Ratio between red and yellow thresholds"
    )
    parser.add_argument(
        "--ema_window", type=int, default=5,
        help="EMA window for collapse detection"
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0.4,
        help="Drop ratio for collapse detection (e.g., 0.4 = 40%% drop)"
    )
    parser.add_argument(
        "--persistence", type=int, default=3,
        help="Number of consecutive eval points for collapse"
    )
    parser.add_argument(
        "--dev_seeds", type=str, default=None,
        help="Comma-separated list of seeds for dev set (e.g., '1,2,3')"
    )
    parser.add_argument(
        "--dev_envs", type=str, default=None,
        help="Comma-separated list of envs for dev set"
    )
    
    args = parser.parse_args()
    
    dev_seeds = None
    if args.dev_seeds:
        dev_seeds = [int(s) for s in args.dev_seeds.split(",")]
    
    dev_envs = None
    if args.dev_envs:
        dev_envs = [e.strip() for e in args.dev_envs.split(",")]
    
    thresholds = calibrate_from_runs(
        runs_dir=args.runs_dir,
        return_key=args.return_key,
        target_fpr=args.target_fpr,
        yellow_red_ratio=args.yellow_red_ratio,
        ema_window=args.ema_window,
        drop_ratio=args.drop_ratio,
        persistence=args.persistence,
        dev_envs=dev_envs,
        dev_seeds=dev_seeds,
    )
    
    print(f"\nCalibrated thresholds:")
    if "tau_yellow" in thresholds:
        print(f"  tau_yellow (rsubt/ewma): {thresholds['tau_yellow']:.4f}")
        print(f"  tau_red (rsubt/ewma): {thresholds['tau_red']:.4f}")
    
    with open(args.output, "w") as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
