"""
Benchmark recipes for RSUBT early-warning evaluation.

Generates commands for running experiments with:
- Calibration set (dev) vs Test set (held-out) splits
- Various stress presets to manufacture collapse
- Proper directory organization for offline scoring
- Support for both gymnasium and EnvPool backends

Usage:
    python -m cleanrl_utils.rsubt.benchmark_recipes \
        --output_dir runs/ \
        --num_seeds 5 \
        --workers 2 \
        --use_envpool  # Optional: use EnvPool for faster training

This will generate a bash script with all commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


# ========================================
# Environment configurations (legacy IDs)
# ========================================
MUJOCO_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Swimmer-v4",
    "InvertedDoublePendulum-v4",
]

DMC_ENVS = [
    "dm_control/cheetah-run-v0",
    "dm_control/walker-walk-v0",
    "dm_control/walker-run-v0",
    "dm_control/hopper-stand-v0",
    "dm_control/hopper-hop-v0",
    "dm_control/humanoid-walk-v0",
    "dm_control/humanoid-run-v0",
    "dm_control/fish-swim-v0",
    "dm_control/acrobot-swingup-v0",
    "dm_control/cartpole-swingup-v0",
]

ATARI_ENVS = [
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
]

# ========================================
# EnvPool-native environment IDs
# ========================================
MUJOCO_ENVS_ENVPOOL = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Swimmer-v4",
    "InvertedDoublePendulum-v4",
]

DMC_ENVS_ENVPOOL = [
    "CheetahRun-v1",
    "WalkerWalk-v1",
    "WalkerRun-v1",
    "HopperStand-v1",
    "HopperHop-v1",
    "HumanoidWalk-v1",
    "HumanoidRun-v1",
    "FishSwim-v1",
    "AcrobotSwingup-v1",
    "CartpoleSwingup-v1",
]

ATARI_ENVS_ENVPOOL = [
    "Breakout-v5",
    "Pong-v5",
    "SpaceInvaders-v5",
    "Seaquest-v5",
    "BeamRider-v5",
    "Enduro-v5",
    "Qbert-v5",
    "MsPacman-v5",
    "Asterix-v5",
    "RoadRunner-v5",
]

# Fragile environments for stress testing (high dimensional, harder to stabilize)
FRAGILE_MUJOCO = ["Humanoid-v4", "Ant-v4", "Walker2d-v4"]
FRAGILE_DMC = ["dm_control/humanoid-walk-v0", "dm_control/humanoid-run-v0", "dm_control/walker-run-v0"]
FRAGILE_DMC_ENVPOOL = ["HumanoidWalk-v1", "HumanoidRun-v1", "WalkerRun-v1"]

# Script mappings for EnvPool
SCRIPT_ENVPOOL_MAP = {
    "ppo_continuous_action.py": "ppo_continuous_action_envpool.py",
    "sac_continuous_action.py": "sac_continuous_action_envpool.py",
    "td3_continuous_action.py": "td3_continuous_action_envpool.py",
    "ppo_atari.py": "ppo_atari_envpool.py",
    "sac_atari.py": "sac_atari_envpool.py",
}

# Stress presets for hyperparameter stress
PPO_STRESS_PRESETS = {
    "value_domination": {
        "vf_coef": 4.0,
        "clip_coef": 0.05,
        "update_epochs": 15,
    },
    "lr_blowup": {
        "learning_rate": "3e-3",
        "update_epochs": 20,
        "num_minibatches": 4,
    },
    "aggressive": {
        "learning_rate": "1e-3",
        "vf_coef": 2.0,
        "clip_coef": 0.1,
        "update_epochs": 15,
    },
}

SAC_STRESS_PRESETS = {
    "high_critic_lr": {
        "q_lr": "3e-3",
    },
    "fast_target": {
        "tau": 0.05,
    },
}

TD3_STRESS_PRESETS = {
    "high_lr": {
        "learning_rate": "1e-3",
    },
    "fast_actor": {
        "policy_frequency": 1,
    },
}

# Environment stress presets (wrapper-based)
ENV_STRESS_PRESETS = {
    "obs_noise": {"obs_noise_std": 0.05},
    "obs_noise_heavy": {"obs_noise_std": 0.15},
    "action_delay_1": {"action_delay": 1},
    "action_delay_2": {"action_delay": 2},
    "reward_scale_high": {"reward_scale": 10.0},
    "reward_scale_low": {"reward_scale": 0.1},
}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    script: str
    envs: list[str]
    total_timesteps: int
    base_args: dict[str, Any]
    stress_args: dict[str, Any] | None = None
    env_stress: dict[str, Any] | None = None
    seeds: list[int] | None = None
    is_dev: bool = False  # True = calibration set, False = test set


def format_args(args: dict[str, Any]) -> str:
    """Format args dict to CLI string."""
    parts = []
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
            else:
                parts.append(f"--no-{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def generate_benchmark_command(
    script: str,
    env_id: str,
    seed: int,
    base_args: dict[str, Any],
    stress_args: dict[str, Any] | None = None,
    env_stress: dict[str, Any] | None = None,
    output_subdir: str = "",
) -> str:
    """Generate a single benchmark command."""
    args = {"env_id": env_id, "seed": seed, **base_args}
    
    if stress_args:
        args.update(stress_args)
    
    if env_stress:
        args.update(env_stress)
    
    cmd = f"python cleanrl/{script} {format_args(args)}"
    
    return cmd


def generate_calibration_test_split(
    num_seeds: int = 5,
    dev_seed_ratio: float = 0.4,
) -> tuple[list[int], list[int]]:
    """
    Generate dev (calibration) and test seed splits.
    
    Args:
        num_seeds: Total number of seeds.
        dev_seed_ratio: Fraction of seeds for dev set.
    
    Returns:
        (dev_seeds, test_seeds)
    """
    all_seeds = list(range(1, num_seeds + 1))
    n_dev = max(1, int(len(all_seeds) * dev_seed_ratio))
    
    # Use first seeds for dev, rest for test
    dev_seeds = all_seeds[:n_dev]
    test_seeds = all_seeds[n_dev:]
    
    return dev_seeds, test_seeds


def generate_full_benchmark_script(
    output_dir: str = "runs",
    num_seeds: int = 5,
    workers: int = 2,
    include_stress: bool = True,
    continuous_timesteps: int = 1000000,
    atari_timesteps: int = 10000000,
    use_envpool: bool = False,
    num_envs: int = 8,
) -> str:
    """
    Generate a full benchmark script with dev/test splits.
    
    Args:
        output_dir: Base output directory.
        num_seeds: Number of seeds per configuration.
        workers: Number of parallel workers.
        include_stress: Whether to include stress test configurations.
        continuous_timesteps: Timesteps for continuous control.
        atari_timesteps: Timesteps for Atari.
        use_envpool: Whether to use EnvPool-native scripts.
        num_envs: Number of parallel environments per run (for EnvPool).
    
    Returns:
        Bash script content.
    """
    dev_seeds, test_seeds = generate_calibration_test_split(num_seeds)
    
    # Select environment lists and script mappings based on backend
    if use_envpool:
        mujoco_envs = MUJOCO_ENVS_ENVPOOL
        dmc_envs = DMC_ENVS_ENVPOOL
        atari_envs = ATARI_ENVS_ENVPOOL
        fragile_mujoco = FRAGILE_MUJOCO  # Same IDs
        fragile_dmc = FRAGILE_DMC_ENVPOOL
        envpool_suffix = "_envpool"
    else:
        mujoco_envs = MUJOCO_ENVS
        dmc_envs = DMC_ENVS
        atari_envs = ATARI_ENVS
        fragile_mujoco = FRAGILE_MUJOCO
        fragile_dmc = FRAGILE_DMC
        envpool_suffix = ""
    
    backend_note = "EnvPool" if use_envpool else "Gymnasium"
    envpool_args = f" --num_envs {num_envs}" if use_envpool else ""
    
    lines = [
        "#!/bin/bash",
        f"# RSUBT Early Warning Benchmark ({backend_note})",
        "# Generated by cleanrl_utils.rsubt.benchmark_recipes",
        "",
        "set -e",
        "",
        f"cd {output_dir.rstrip('/')}/.. && export MUJOCO_GL=osmesa",
        "",
        f"DEV_SEEDS='{' '.join(map(str, dev_seeds))}'",
        f"TEST_SEEDS='{' '.join(map(str, test_seeds))}'",
        "",
        "# Common args",
        f"COMMON_ARGS='--rsubt_monitor --eval_interval_steps 10000 --no-anneal_lr{envpool_args}'",
        "",
    ]
    
    # Helper function to add benchmark block
    def add_benchmark_block(
        name: str,
        envs: list[str],
        script: str,
        timesteps: int,
        extra_args: str = "",
        stress_name: str = "",
        seeds_var: str = "DEV_SEEDS",
        output_subdir: str = "calib",
    ):
        lines.append(f"# === {name} ===")
        env_list = " ".join(envs)
        
        cmd = f"""uv run python -m cleanrl_utils.benchmark \\
  --env-ids {env_list} \\
  --command "python cleanrl/{script} $COMMON_ARGS --total_timesteps {timesteps} {extra_args}" \\
  --num-seeds 1 --workers {workers}"""
        
        # Run for each seed
        lines.append(f"for seed in ${seeds_var}; do")
        lines.append(f'  echo "Running {name} seed $seed"')
        lines.append(f"  {cmd} --start-seed $seed")
        lines.append("done")
        lines.append("")
    
    # ========== CALIBRATION SET (dev) ==========
    lines.append("# ========================================")
    lines.append("# CALIBRATION SET (dev)")
    lines.append("# ========================================")
    lines.append("")
    
    # Baseline runs (no stress) - for calibrating thresholds on stable runs
    add_benchmark_block(
        "PPO MuJoCo Baseline (dev)",
        mujoco_envs,
        f"ppo_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--learning_rate 3e-4",
        seeds_var="DEV_SEEDS",
    )
    
    add_benchmark_block(
        "SAC MuJoCo Baseline (dev)",
        mujoco_envs,
        f"sac_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--policy_lr 3e-4 --q_lr 3e-4",
        seeds_var="DEV_SEEDS",
    )
    
    add_benchmark_block(
        "TD3 MuJoCo Baseline (dev)",
        mujoco_envs,
        f"td3_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--learning_rate 3e-4",
        seeds_var="DEV_SEEDS",
    )
    
    if include_stress:
        # Stress runs (dev) - to get some collapses for calibration
        lines.append("# --- Stress runs (dev) ---")
        
        add_benchmark_block(
            "PPO MuJoCo Stress:value_domination (dev)",
            fragile_mujoco,
            f"ppo_continuous_action{envpool_suffix}.py",
            continuous_timesteps,
            "--learning_rate 3e-4 --vf_coef 4.0 --clip_coef 0.05 --update_epochs 15",
            seeds_var="DEV_SEEDS",
        )
        
        add_benchmark_block(
            "PPO MuJoCo Stress:obs_noise (dev)",
            fragile_mujoco,
            f"ppo_continuous_action{envpool_suffix}.py",
            continuous_timesteps,
            "--learning_rate 3e-4 --obs_noise_std 0.1",
            seeds_var="DEV_SEEDS",
        )
    
    # ========== TEST SET ==========
    lines.append("# ========================================")
    lines.append("# TEST SET (held-out)")
    lines.append("# ========================================")
    lines.append("")
    
    # Baseline runs (test)
    add_benchmark_block(
        "PPO MuJoCo Baseline (test)",
        mujoco_envs,
        f"ppo_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--learning_rate 3e-4",
        seeds_var="TEST_SEEDS",
    )
    
    add_benchmark_block(
        "SAC MuJoCo Baseline (test)",
        mujoco_envs,
        f"sac_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--policy_lr 3e-4 --q_lr 3e-4",
        seeds_var="TEST_SEEDS",
    )
    
    add_benchmark_block(
        "TD3 MuJoCo Baseline (test)",
        mujoco_envs,
        f"td3_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--learning_rate 3e-4",
        seeds_var="TEST_SEEDS",
    )
    
    if include_stress:
        lines.append("# --- Stress runs (test) ---")
        
        # Multiple stress configurations for test set
        stress_configs = [
            ("value_domination", "--vf_coef 4.0 --clip_coef 0.05 --update_epochs 15"),
            ("lr_blowup", "--learning_rate 3e-3 --update_epochs 20 --num_minibatches 4"),
            ("obs_noise", "--obs_noise_std 0.1"),
            ("action_delay", "--action_delay 2"),
        ]
        
        for stress_name, stress_args in stress_configs:
            add_benchmark_block(
                f"PPO MuJoCo Stress:{stress_name} (test)",
                fragile_mujoco,
                f"ppo_continuous_action{envpool_suffix}.py",
                continuous_timesteps,
                f"--learning_rate 3e-4 {stress_args}",
                seeds_var="TEST_SEEDS",
            )
        
        # SAC stress
        add_benchmark_block(
            "SAC MuJoCo Stress:high_q_lr (test)",
            fragile_mujoco,
            f"sac_continuous_action{envpool_suffix}.py",
            continuous_timesteps,
            "--policy_lr 3e-4 --q_lr 3e-3",
            seeds_var="TEST_SEEDS",
        )
        
        # TD3 stress
        add_benchmark_block(
            "TD3 MuJoCo Stress:high_lr (test)",
            fragile_mujoco,
            f"td3_continuous_action{envpool_suffix}.py",
            continuous_timesteps,
            "--learning_rate 1e-3",
            seeds_var="TEST_SEEDS",
        )
    
    # ========== RSUBT CONTROL EXPERIMENTS ==========
    lines.append("# ========================================")
    lines.append("# RSUBT CONTROL (mitigation) EXPERIMENTS")
    lines.append("# ========================================")
    lines.append("")
    
    # Same stress configs but with rsubt_control enabled
    add_benchmark_block(
        "PPO MuJoCo Stress+Control:value_domination",
        fragile_mujoco,
        f"ppo_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--learning_rate 3e-4 --vf_coef 4.0 --clip_coef 0.05 --update_epochs 15 --rsubt_control",
        seeds_var="TEST_SEEDS",
    )
    
    add_benchmark_block(
        "PPO MuJoCo Stress+Control:obs_noise",
        fragile_mujoco,
        f"ppo_continuous_action{envpool_suffix}.py",
        continuous_timesteps,
        "--learning_rate 3e-4 --obs_noise_std 0.1 --rsubt_control",
        seeds_var="TEST_SEEDS",
    )
    
    lines.append("")
    lines.append("echo '=== ALL BENCHMARK RUNS COMPLETE ==='")
    lines.append("")
    
    # Add calibration and evaluation commands
    lines.extend([
        "# ========================================",
        "# POST-PROCESSING",
        "# ========================================",
        "",
        "# Calibrate thresholds from dev runs",
        f"# python -m cleanrl_utils.rsubt.calibrate_thresholds \\",
        f"#   --runs_dir {output_dir}/ \\",
        f"#   --dev_seeds '{','.join(map(str, dev_seeds))}' \\",
        "#   --output thresholds.json",
        "",
        "# Evaluate on test runs",
        f"# python -m cleanrl_utils.rsubt.evaluate_early_warning \\",
        f"#   --runs_dir {output_dir}/ \\",
        "#   --thresholds thresholds.json \\",
        "#   --output results.json",
        "",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate RSUBT benchmark recipes"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark.sh",
        help="Output script file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs",
        help="Base directory for runs"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5,
        help="Number of seeds per configuration"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--no_stress", action="store_true",
        help="Skip stress test configurations"
    )
    parser.add_argument(
        "--continuous_timesteps", type=int, default=1000000,
        help="Timesteps for continuous control"
    )
    parser.add_argument(
        "--atari_timesteps", type=int, default=10000000,
        help="Timesteps for Atari"
    )
    parser.add_argument(
        "--use_envpool", action="store_true",
        help="Use EnvPool-native scripts for faster training"
    )
    parser.add_argument(
        "--num_envs", type=int, default=8,
        help="Number of parallel environments per run (for EnvPool)"
    )
    
    args = parser.parse_args()
    
    script = generate_full_benchmark_script(
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
        workers=args.workers,
        include_stress=not args.no_stress,
        continuous_timesteps=args.continuous_timesteps,
        atari_timesteps=args.atari_timesteps,
        use_envpool=args.use_envpool,
        num_envs=args.num_envs,
    )
    
    with open(args.output, "w") as f:
        f.write(script)
    
    print(f"Benchmark script written to {args.output}")
    print(f"Run with: bash {args.output}")


if __name__ == "__main__":
    main()
