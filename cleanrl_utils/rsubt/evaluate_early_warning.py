"""
Evaluate early warning performance of Rsubt and baseline metrics.

Reads TensorBoard event files and computes:
- Collapse labels (binary) using EMA-based definition
- Alarm times for each metric
- Lead time (how early the alarm fires before collapse)
- TPR@FPR (true positive rate at fixed false positive rate)
- AUROC and AUPRC for "collapse within H steps" prediction

Usage:
    python -m cleanrl_utils.rsubt.evaluate_early_warning \
        --runs_dir runs/test/ \
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


def compute_ema(values: list[float], alpha: float = 0.2) -> list[float]:
    """Compute exponential moving average."""
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
    Detect collapse time using EMA-based definition.
    
    Collapse is defined as:
    - EMA(R(t)) <= (1 - drop_ratio) * R_best(t) for `persistence` consecutive evals.
    """
    if len(returns) < persistence:
        return None
    
    alpha = 2.0 / (ema_window + 1)
    ema = compute_ema(returns, alpha)
    
    r_best = ema[0]
    low_count = 0
    
    for i, (step, ema_val) in enumerate(zip(steps, ema)):
        if ema_val > r_best:
            r_best = ema_val
            low_count = 0
        else:
            if r_best > 0:
                threshold = (1 - drop_ratio) * r_best
                is_collapsed = ema_val < threshold
            elif r_best < 0:
                threshold = (1 + drop_ratio) * r_best
                is_collapsed = ema_val < threshold
            else:
                threshold = -abs(drop_ratio)
                is_collapsed = ema_val < threshold
            
            if is_collapsed:
                low_count += 1
                if low_count >= persistence:
                    collapse_idx = i - persistence + 1
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
    """Compute lead time in steps (positive = early warning)."""
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
        steps = run.get(f"{metric_key}_steps", [])
        values = run.get(f"{metric_key}_values", [])
        alarm_step = detect_alarm(steps, values, threshold, consecutive, higher_is_alarm)
        
        if alarm_step is not None and run["collapse_step"] is not None:
            lead = compute_lead_time(run["collapse_step"], alarm_step)
            if lead is not None and lead > 0:
                true_positives += 1
                lead_times.append(lead)
    
    # False positives: alarm in non-collapse runs
    false_positives = 0
    for run in non_collapse_runs:
        steps = run.get(f"{metric_key}_steps", [])
        values = run.get(f"{metric_key}_values", [])
        alarm_step = detect_alarm(steps, values, threshold, consecutive, higher_is_alarm)
        if alarm_step is not None:
            false_positives += 1
    
    # Compute rates
    n_collapse = len(collapse_runs)
    n_non_collapse = len(non_collapse_runs)
    
    tpr = true_positives / n_collapse if n_collapse > 0 else 0.0
    fpr = false_positives / n_non_collapse if n_non_collapse > 0 else 0.0
    
    # Lead time statistics
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    std_lead = float(np.std(lead_times)) if lead_times else 0.0
    
    return {
        "tpr": tpr,
        "fpr": fpr,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "n_collapse_runs": n_collapse,
        "n_non_collapse_runs": n_non_collapse,
        "mean_lead_time": mean_lead,
        "median_lead_time": median_lead,
        "std_lead_time": std_lead,
        "lead_times": lead_times,
    }


def compute_auc_roc(
    runs: list[dict],
    metric_key: str,
    horizon_steps: int = 50000,
    higher_is_risk: bool = True,
) -> float:
    """
    Compute AUROC for "collapse within horizon" prediction.
    
    For each time point, treats the metric value as a score predicting
    whether collapse occurs within the next `horizon_steps`.
    
    Args:
        runs: List of run data dicts.
        metric_key: Key for the metric.
        horizon_steps: Prediction horizon.
        higher_is_risk: If True, higher values predict collapse.
    
    Returns:
        AUROC score.
    """
    all_scores = []
    all_labels = []
    
    for run in runs:
        steps = run.get(f"{metric_key}_steps", [])
        values = run.get(f"{metric_key}_values", [])
        collapse_step = run["collapse_step"]
        
        for step, val in zip(steps, values):
            # Label: will collapse within horizon?
            if collapse_step is not None:
                label = 1 if 0 < (collapse_step - step) <= horizon_steps else 0
            else:
                label = 0
            
            all_scores.append(val if higher_is_risk else -val)
            all_labels.append(label)
    
    if not all_scores or sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        return 0.5  # No valid AUC
    
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(all_labels, all_scores))
    except ImportError:
        # Fallback: Mann-Whitney U statistic
        try:
            from scipy import stats
            positive_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
            negative_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]
            if not positive_scores or not negative_scores:
                return 0.5
            statistic, _ = stats.mannwhitneyu(positive_scores, negative_scores, alternative="greater")
            return float(statistic / (len(positive_scores) * len(negative_scores)))
        except Exception:
            return 0.5


def compute_auc_pr(
    runs: list[dict],
    metric_key: str,
    horizon_steps: int = 50000,
    higher_is_risk: bool = True,
) -> float:
    """
    Compute AUPRC (Area Under Precision-Recall Curve) for collapse prediction.
    
    Args:
        runs: List of run data dicts.
        metric_key: Key for the metric.
        horizon_steps: Prediction horizon.
        higher_is_risk: If True, higher values predict collapse.
    
    Returns:
        AUPRC score.
    """
    all_scores = []
    all_labels = []
    
    for run in runs:
        steps = run.get(f"{metric_key}_steps", [])
        values = run.get(f"{metric_key}_values", [])
        collapse_step = run["collapse_step"]
        
        for step, val in zip(steps, values):
            if collapse_step is not None:
                label = 1 if 0 < (collapse_step - step) <= horizon_steps else 0
            else:
                label = 0
            
            all_scores.append(val if higher_is_risk else -val)
            all_labels.append(label)
    
    if not all_scores or sum(all_labels) == 0 or sum(all_labels) == len(all_labels):
        return 0.0
    
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(all_labels, all_scores))
    except ImportError:
        # Simple approximation
        return 0.0


def evaluate_all_metrics(
    runs_dir: str | Path,
    thresholds: dict[str, Any],
    metrics_config: dict[str, dict] | None = None,
    return_key: str = "eval/return_mean",
    ema_window: int = 5,
    drop_ratio: float = 0.4,
    persistence: int = 3,
    horizon_steps: int = 50000,
) -> dict[str, Any]:
    """
    Evaluate all metrics across all runs.
    
    Args:
        runs_dir: Directory containing run subdirectories.
        thresholds: Dict with threshold values from calibration.
        metrics_config: Configuration for each metric (key, threshold, direction).
        return_key: TensorBoard key for returns.
        ema_window: EMA window for collapse detection.
        drop_ratio: Drop ratio for collapse detection.
        persistence: Persistence for collapse detection.
        horizon_steps: Prediction horizon for AUC computation.
    
    Returns:
        Results dictionary.
    """
    runs_dir = Path(runs_dir)
    
    # Build metrics config from thresholds
    if metrics_config is None:
        metrics_config = {}
        
        # Add Rsubt metrics
        if "metrics" in thresholds:
            for metric_name, metric_thresholds in thresholds["metrics"].items():
                # Determine direction based on metric name
                higher_is_alarm = True
                if "entropy" in metric_name.lower() or "stablerank" in metric_name.lower():
                    higher_is_alarm = False
                
                metrics_config[metric_name] = {
                    "threshold": metric_thresholds["tau_yellow"],
                    "higher_is_alarm": higher_is_alarm,
                }
        else:
            # Legacy format
            tau_y = thresholds.get("tau_yellow", 0.5)
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
        return_steps, return_values = load_scalar_from_events(run_dir, return_key)
        if not return_values:
            # Fallback to training returns
            return_steps, return_values = load_scalar_from_events(run_dir, "charts/episodic_return")
        
        run_data["collapse_step"] = detect_collapse_ema(
            return_steps, return_values,
            ema_window=ema_window,
            drop_ratio=drop_ratio,
            persistence=persistence,
        )
        
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
        
        # Compute AUROC
        try:
            auroc = compute_auc_roc(
                runs, metric_key,
                horizon_steps=horizon_steps,
                higher_is_risk=config["higher_is_alarm"],
            )
            metric_results["auroc"] = auroc
        except Exception:
            metric_results["auroc"] = None
        
        # Compute AUPRC
        try:
            auprc = compute_auc_pr(
                runs, metric_key,
                horizon_steps=horizon_steps,
                higher_is_risk=config["higher_is_alarm"],
            )
            metric_results["auprc"] = auprc
        except Exception:
            metric_results["auprc"] = None
        
        results["metrics"][metric_key] = metric_results
        
        print(f"  TPR: {metric_results['tpr']:.3f}")
        print(f"  FPR: {metric_results['fpr']:.3f}")
        print(f"  Mean lead time: {metric_results['mean_lead_time']:.0f} steps")
        if metric_results.get("auroc") is not None:
            print(f"  AUROC: {metric_results['auroc']:.3f}")
        if metric_results.get("auprc") is not None:
            print(f"  AUPRC: {metric_results['auprc']:.3f}")
    
    # Summary statistics
    results["summary"] = {
        "n_runs": len(runs),
        "n_collapse_runs": n_collapse,
        "n_non_collapse_runs": len(runs) - n_collapse,
        "horizon_steps": horizon_steps,
        "collapse_config": {
            "ema_window": ema_window,
            "drop_ratio": drop_ratio,
            "persistence": persistence,
        },
    }
    
    # Ranking of metrics by AUROC
    if any(m.get("auroc") is not None for m in results["metrics"].values()):
        ranking = sorted(
            [(k, v.get("auroc", 0)) for k, v in results["metrics"].items()],
            key=lambda x: x[1] if x[1] is not None else 0,
            reverse=True,
        )
        print("\n=== Metric Ranking by AUROC ===")
        for i, (metric, auroc) in enumerate(ranking, 1):
            if auroc is not None:
                print(f"  {i}. {metric}: {auroc:.3f}")
        results["ranking_auroc"] = ranking
    
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
    parser.add_argument(
        "--return_key", type=str, default="eval/return_mean",
        help="TensorBoard key for returns"
    )
    parser.add_argument(
        "--ema_window", type=int, default=5,
        help="EMA window for collapse detection"
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0.4,
        help="Drop ratio for collapse detection"
    )
    parser.add_argument(
        "--persistence", type=int, default=3,
        help="Persistence for collapse detection"
    )
    
    args = parser.parse_args()
    
    # Load thresholds
    if args.thresholds and Path(args.thresholds).exists():
        with open(args.thresholds) as f:
            thresholds = json.load(f)
    else:
        print("No thresholds file provided, using defaults")
        thresholds = {"tau_yellow": 0.5, "tau_red": 1.0}
    
    results = evaluate_all_metrics(
        args.runs_dir,
        thresholds,
        return_key=args.return_key,
        ema_window=args.ema_window,
        drop_ratio=args.drop_ratio,
        persistence=args.persistence,
        horizon_steps=args.horizon,
    )
    
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
