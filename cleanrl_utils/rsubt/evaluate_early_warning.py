"""
Evaluate early warning performance of Rsubt and baseline metrics.

Reads TensorBoard event files and computes:
- Collapse labels (binary)
- Alarm times for each metric
- Lead time (how early the alarm fires before collapse)
- TPR@FPR (true positive rate at fixed false positive rate)
- AUC for "collapse within H steps" prediction

Usage:
    python -m cleanrl_utils.rsubt.evaluate_early_warning \
        --runs_dir runs/evaluation/ \
        --thresholds thresholds.json \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise ImportError("tensorboard is required. Install with: pip install tensorboard")


@dataclass
class RunMetrics:
    """Metrics for a single run."""
    run_name: str
    collapse_step: int | None
    alarm_times: dict[str, int | None]  # metric_name -> alarm step
    lead_times: dict[str, float | None]  # metric_name -> lead time in steps


def load_scalar_from_events(
    logdir: str | Path,
    scalar_key: str,
) -> tuple[list[int], list[float]]:
    """Load a scalar time series from TensorBoard event files."""
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    
    if scalar_key not in ea.Tags().get("scalars", []):
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
    """Detect collapse time from return history."""
    if len(returns) < consecutive_evals:
        return None
    
    max_return = float("-inf")
    low_count = 0
    
    for i, (step, ret) in enumerate(zip(steps, returns)):
        if ret > max_return:
            max_return = ret
            low_count = 0
        else:
            threshold = threshold_ratio * max_return
            if max_return > 0 and ret < threshold:
                low_count += 1
                if low_count >= consecutive_evals:
                    collapse_idx = i - consecutive_evals + 1
                    return steps[collapse_idx]
            else:
                low_count = 0
    
    return None


def detect_alarm(
    steps: list[int],
    values: list[float],
    threshold: float,
    consecutive: int = 2,
    higher_is_alarm: bool = True,
) -> int | None:
    """
    Detect when a metric crosses the alarm threshold.
    
    Args:
        steps: Time steps.
        values: Metric values.
        threshold: Alarm threshold.
        consecutive: Number of consecutive crossings needed.
        higher_is_alarm: If True, alarm when value > threshold.
    
    Returns:
        Alarm step or None.
    """
    if not values:
        return None
    
    count = 0
    for i, (step, val) in enumerate(zip(steps, values)):
        if higher_is_alarm:
            triggered = val > threshold
        else:
            triggered = val < threshold
        
        if triggered:
            count += 1
            if count >= consecutive:
                alarm_idx = i - consecutive + 1
                return steps[alarm_idx]
        else:
            count = 0
    
    return None


def compute_lead_time(
    collapse_step: int | None,
    alarm_step: int | None,
) -> float | None:
    """Compute lead time in steps."""
    if collapse_step is None or alarm_step is None:
        return None
    
    lead_time = collapse_step - alarm_step
    if lead_time < 0:
        return None  # Alarm after collapse is not useful
    
    return lead_time


def evaluate_metric(
    runs: list[dict],
    metric_key: str,
    threshold: float,
    consecutive: int = 2,
    higher_is_alarm: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a single metric's early warning performance.
    
    Args:
        runs: List of run data dicts with collapse info and metric values.
        metric_key: Key for the metric to evaluate.
        threshold: Alarm threshold.
        consecutive: Consecutive crossings needed for alarm.
        higher_is_alarm: Direction of alarm.
    
    Returns:
        Dictionary with TPR, lead times, etc.
    """
    collapse_runs = [r for r in runs if r["collapse_step"] is not None]
    non_collapse_runs = [r for r in runs if r["collapse_step"] is None]
    
    # True positives: alarm before collapse
    true_positives = 0
    lead_times = []
    
    for run in collapse_runs:
        steps, values = run.get(f"{metric_key}_steps", []), run.get(f"{metric_key}_values", [])
        alarm_step = detect_alarm(steps, values, threshold, consecutive, higher_is_alarm)
        
        if alarm_step is not None and run["collapse_step"] is not None:
            lead = compute_lead_time(run["collapse_step"], alarm_step)
            if lead is not None and lead > 0:
                true_positives += 1
                lead_times.append(lead)
    
    # False positives: alarm in non-collapse runs
    false_positives = 0
    for run in non_collapse_runs:
        steps, values = run.get(f"{metric_key}_steps", []), run.get(f"{metric_key}_values", [])
        alarm_step = detect_alarm(steps, values, threshold, consecutive, higher_is_alarm)
        if alarm_step is not None:
            false_positives += 1
    
    # Compute rates
    n_collapse = len(collapse_runs)
    n_non_collapse = len(non_collapse_runs)
    
    tpr = true_positives / n_collapse if n_collapse > 0 else 0.0
    fpr = false_positives / n_non_collapse if n_non_collapse > 0 else 0.0
    
    # Lead time statistics
    mean_lead = np.mean(lead_times) if lead_times else 0.0
    median_lead = np.median(lead_times) if lead_times else 0.0
    
    return {
        "tpr": tpr,
        "fpr": fpr,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "n_collapse_runs": n_collapse,
        "n_non_collapse_runs": n_non_collapse,
        "mean_lead_time": mean_lead,
        "median_lead_time": median_lead,
        "lead_times": lead_times,
    }


def compute_auc_collapse_prediction(
    runs: list[dict],
    metric_key: str,
    horizon_steps: int = 50000,
    higher_is_risk: bool = True,
) -> float:
    """
    Compute AUC for "collapse within horizon" prediction.
    
    For each time point, treats the metric value as a score predicting
    whether collapse occurs within the next `horizon_steps`.
    
    Args:
        runs: List of run data dicts.
        metric_key: Key for the metric.
        horizon_steps: Prediction horizon.
        higher_is_risk: If True, higher values predict collapse.
    
    Returns:
        AUC score.
    """
    all_scores = []
    all_labels = []
    
    for run in runs:
        steps, values = run.get(f"{metric_key}_steps", []), run.get(f"{metric_key}_values", [])
        collapse_step = run["collapse_step"]
        
        for i, (step, val) in enumerate(zip(steps, values)):
            # Label: will collapse within horizon?
            if collapse_step is not None:
                label = 1 if (collapse_step - step) <= horizon_steps and (collapse_step - step) > 0 else 0
            else:
                label = 0
            
            all_scores.append(val if higher_is_risk else -val)
            all_labels.append(label)
    
    if not all_scores or sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        return 0.5  # No valid AUC
    
    # Simple AUC calculation via Mann-Whitney U statistic
    from scipy import stats
    
    positive_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
    negative_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]
    
    if not positive_scores or not negative_scores:
        return 0.5
    
    try:
        statistic, _ = stats.mannwhitneyu(positive_scores, negative_scores, alternative="greater")
        auc = statistic / (len(positive_scores) * len(negative_scores))
    except Exception:
        auc = 0.5
    
    return float(auc)


def evaluate_all_metrics(
    runs_dir: str | Path,
    thresholds: dict[str, float],
    metrics_config: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """
    Evaluate all metrics across all runs.
    
    Args:
        runs_dir: Directory containing run subdirectories.
        thresholds: Dict with threshold values.
        metrics_config: Configuration for each metric (key, threshold, direction).
    
    Returns:
        Results dictionary.
    """
    runs_dir = Path(runs_dir)
    
    # Default metrics to evaluate
    if metrics_config is None:
        tau_y = thresholds.get("tau_yellow", 0.5)
        tau_r = thresholds.get("tau_red", 1.0)
        
        metrics_config = {
            "rsubt/ewma": {"threshold": tau_y, "higher_is_alarm": True},
            "rsubt/shock": {"threshold": tau_y, "higher_is_alarm": True},
            "losses/approx_kl": {"threshold": 0.02, "higher_is_alarm": True},
            "losses/value_loss": {"threshold": 100.0, "higher_is_alarm": True},
            "losses/entropy": {"threshold": 0.1, "higher_is_alarm": False},
            "baseline/grad_norm_actor": {"threshold": 10.0, "higher_is_alarm": True},
            "reprank/actor_stablerank": {"threshold": 5.0, "higher_is_alarm": False},
        }
    
    # Load all runs
    runs = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_data = {"name": run_dir.name}
        
        # Load returns and detect collapse
        return_steps, return_values = load_scalar_from_events(run_dir, "charts/episodic_return")
        run_data["collapse_step"] = detect_collapse(return_steps, return_values)
        
        # Load all metrics
        for metric_key in metrics_config.keys():
            steps, values = load_scalar_from_events(run_dir, metric_key)
            run_data[f"{metric_key}_steps"] = steps
            run_data[f"{metric_key}_values"] = values
        
        runs.append(run_data)
    
    print(f"Loaded {len(runs)} runs")
    n_collapse = sum(1 for r in runs if r["collapse_step"] is not None)
    print(f"  {n_collapse} collapse runs")
    print(f"  {len(runs) - n_collapse} non-collapse runs")
    
    # Evaluate each metric
    results = {"metrics": {}}
    
    for metric_key, config in metrics_config.items():
        print(f"\nEvaluating {metric_key}...")
        
        metric_results = evaluate_metric(
            runs,
            metric_key,
            threshold=config["threshold"],
            higher_is_alarm=config["higher_is_alarm"],
        )
        
        # Compute AUC
        try:
            auc = compute_auc_collapse_prediction(
                runs, metric_key,
                higher_is_risk=config["higher_is_alarm"],
            )
            metric_results["auc"] = auc
        except ImportError:
            metric_results["auc"] = None
            print("  Warning: scipy not available for AUC computation")
        
        results["metrics"][metric_key] = metric_results
        
        print(f"  TPR: {metric_results['tpr']:.3f}")
        print(f"  FPR: {metric_results['fpr']:.3f}")
        print(f"  Mean lead time: {metric_results['mean_lead_time']:.0f}")
        if metric_results.get("auc") is not None:
            print(f"  AUC: {metric_results['auc']:.3f}")
    
    # Summary statistics
    results["summary"] = {
        "n_runs": len(runs),
        "n_collapse_runs": n_collapse,
        "n_non_collapse_runs": len(runs) - n_collapse,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate early warning performance of Rsubt and baselines"
    )
    parser.add_argument(
        "--runs_dir", type=str, required=True,
        help="Directory containing TensorBoard run subdirectories"
    )
    parser.add_argument(
        "--thresholds", type=str, default=None,
        help="JSON file with calibrated thresholds (optional)"
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--horizon", type=int, default=50000,
        help="Prediction horizon in steps for AUC computation"
    )
    
    args = parser.parse_args()
    
    # Load thresholds
    if args.thresholds and Path(args.thresholds).exists():
        with open(args.thresholds) as f:
            thresholds = json.load(f)
    else:
        print("No thresholds file provided, using defaults")
        thresholds = {"tau_yellow": 0.5, "tau_red": 1.0}
    
    results = evaluate_all_metrics(args.runs_dir, thresholds)
    
    # Save results
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    
    results = convert_for_json(results)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
