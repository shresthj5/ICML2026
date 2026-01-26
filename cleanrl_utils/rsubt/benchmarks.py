"""
Benchmark command recipes for Rsubt experiments.

Provides pre-configured benchmark commands for:
- MuJoCo continuous control (PPO, SAC)
- DeepMind Control Suite (PPO, SAC)
- Atari (PPO, SAC)
- Stress scenarios (S1: KL-looks-fine, S3: scaling brittleness)

Usage:
    python -m cleanrl_utils.rsubt.benchmarks --suite mujoco --algo ppo
    python -m cleanrl_utils.rsubt.benchmarks --suite all --output benchmark_commands.sh
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


# ============================================================================
# Environment configurations
# ============================================================================

MUJOCO_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Swimmer-v4",
    "Reacher-v4",
]

MUJOCO_HARD_ENVS = ["Humanoid-v4", "Ant-v4", "Walker2d-v4"]

DMC_ENVS = [
    "dm_control/cheetah-run-v0",
    "dm_control/walker-walk-v0",
    "dm_control/walker-run-v0",
    "dm_control/hopper-hop-v0",
    "dm_control/humanoid-walk-v0",
    "dm_control/humanoid-run-v0",
    "dm_control/quadruped-walk-v0",
    "dm_control/quadruped-run-v0",
    "dm_control/finger-turn_hard-v0",
    "dm_control/fish-swim-v0",
]

DMC_HARD_ENVS = [
    "dm_control/humanoid-run-v0",
    "dm_control/quadruped-run-v0",
    "dm_control/finger-turn_hard-v0",
]

ATARI_ENVS = [
    "BreakoutNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "PongNoFrameskip-v4",
]

ATARI_HARD_ENVS = [
    "SeaquestNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
]


# ============================================================================
# Stress scenario configurations
# ============================================================================

@dataclass
class StressConfig:
    """Configuration for a stress scenario."""
    name: str
    description: str
    extra_args: dict[str, Any]


# S1: KL looks fine, but critic instability causes collapse
STRESS_S1_PPO = StressConfig(
    name="s1_critic_instability",
    description="PPO with small clip_coef and destabilized critic",
    extra_args={
        "clip_coef": 0.1,
        "target_kl": 0.01,  # Tight KL constraint
        "vf_coef": 2.0,     # High value function weight
        "update_epochs": 15, # More epochs
        "num_minibatches": 64,  # Smaller minibatches
    },
)

# S3: Scaling-induced brittleness (requires network width modification)
# Note: This requires custom network architecture support
STRESS_S3_WIDTHS = [64, 256, 1024]


# ============================================================================
# Command generation
# ============================================================================

def generate_benchmark_command(
    script: str,
    env_id: str,
    seed: int,
    rsubt_enabled: bool = True,
    rsubt_control: bool = False,
    extra_args: dict[str, Any] | None = None,
    use_uv: bool = True,
) -> str:
    """Generate a single benchmark command."""
    if use_uv:
        cmd = f"uv run python cleanrl/{script}"
    else:
        cmd = f"python cleanrl/{script}"
    
    cmd += f" --env_id {env_id}"
    cmd += f" --seed {seed}"
    
    if rsubt_enabled:
        cmd += " --rsubt_monitor"
        if rsubt_control:
            cmd += " --rsubt_control"
    
    if extra_args:
        for key, value in extra_args.items():
            if isinstance(value, bool):
                if value:
                    cmd += f" --{key}"
            else:
                cmd += f" --{key} {value}"
    
    return cmd


def generate_suite_commands(
    suite: str,
    algo: str,
    seeds: list[int] = [1, 2, 3, 4, 5],
    stress_scenario: str | None = None,
    rsubt_control: bool = False,
) -> list[str]:
    """Generate commands for a benchmark suite."""
    commands = []
    
    # Determine script and environments
    if algo == "ppo":
        if suite == "mujoco":
            script = "ppo_continuous_action.py"
            envs = MUJOCO_ENVS
        elif suite == "dmc":
            script = "ppo_continuous_action.py"
            envs = DMC_ENVS
        elif suite == "atari":
            script = "ppo_atari.py"
            envs = ATARI_ENVS
        else:
            raise ValueError(f"Unknown suite: {suite}")
    elif algo == "sac":
        if suite == "mujoco":
            script = "sac_continuous_action.py"
            envs = MUJOCO_ENVS
        elif suite == "dmc":
            script = "sac_continuous_action.py"
            envs = DMC_ENVS
        elif suite == "atari":
            script = "sac_atari.py"
            envs = ATARI_ENVS
        else:
            raise ValueError(f"Unknown suite: {suite}")
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Apply stress scenario configuration
    extra_args = {}
    if stress_scenario == "s1" and algo == "ppo":
        extra_args = STRESS_S1_PPO.extra_args.copy()
    
    # Generate commands
    for env_id in envs:
        for seed in seeds:
            cmd = generate_benchmark_command(
                script=script,
                env_id=env_id,
                seed=seed,
                rsubt_enabled=True,
                rsubt_control=rsubt_control,
                extra_args=extra_args if extra_args else None,
            )
            commands.append(cmd)
    
    return commands


def generate_benchmark_py_config(
    suite: str,
    algo: str,
    seeds: list[int] = [1, 2, 3, 4, 5],
    stress_scenario: str | None = None,
) -> str:
    """
    Generate configuration for cleanrl_utils.benchmark.
    
    Returns a command that can be used with:
        python -m cleanrl_utils.benchmark --command "..."
    """
    # Determine script and environments
    if algo == "ppo":
        if suite == "mujoco":
            script = "ppo_continuous_action.py"
            envs = MUJOCO_ENVS
        elif suite == "dmc":
            script = "ppo_continuous_action.py"
            envs = DMC_ENVS
        elif suite == "atari":
            script = "ppo_atari.py"
            envs = ATARI_ENVS
        else:
            raise ValueError(f"Unknown suite: {suite}")
    elif algo == "sac":
        if suite == "mujoco":
            script = "sac_continuous_action.py"
            envs = MUJOCO_ENVS
        elif suite == "dmc":
            script = "sac_continuous_action.py"
            envs = DMC_ENVS
        elif suite == "atari":
            script = "sac_atari.py"
            envs = ATARI_ENVS
        else:
            raise ValueError(f"Unknown suite: {suite}")
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Build extra args string
    extra_args_str = "--rsubt_monitor"
    if stress_scenario == "s1" and algo == "ppo":
        for key, value in STRESS_S1_PPO.extra_args.items():
            extra_args_str += f" --{key} {value}"
    
    # Format for benchmark utility
    env_ids_str = " ".join(envs)
    seeds_str = " ".join(str(s) for s in seeds)
    
    cmd = f'''python -m cleanrl_utils.benchmark \\
    --env-ids {env_ids_str} \\
    --command "python cleanrl/{script} --track --env-id {{env_id}} --seed {{seed}} {extra_args_str}" \\
    --num-seeds {len(seeds)} \\
    --workers 4'''
    
    return cmd


def print_all_benchmarks():
    """Print all benchmark configurations."""
    print("=" * 80)
    print("Rsubt Experiment Benchmark Commands")
    print("=" * 80)
    
    # Normal training
    print("\n### NORMAL TRAINING (5 seeds) ###\n")
    
    for suite in ["mujoco", "dmc", "atari"]:
        for algo in ["ppo", "sac"]:
            print(f"\n## {suite.upper()} - {algo.upper()} ##")
            print(generate_benchmark_py_config(suite, algo))
    
    # Stress scenarios
    print("\n" + "=" * 80)
    print("### STRESS SCENARIO S1 (10 seeds on hard envs) ###")
    print("=" * 80)
    
    print("\n## MuJoCo - PPO S1 (hard envs only) ##")
    cmds = generate_suite_commands("mujoco", "ppo", seeds=list(range(1, 11)), stress_scenario="s1")
    # Filter to hard envs only
    hard_cmds = [c for c in cmds if any(e in c for e in MUJOCO_HARD_ENVS)]
    for cmd in hard_cmds[:3]:  # Show first 3 as example
        print(f"  {cmd}")
    print(f"  ... ({len(hard_cmds)} total commands)")
    
    # Mitigation experiments
    print("\n" + "=" * 80)
    print("### MITIGATION EXPERIMENTS (with --rsubt_control) ###")
    print("=" * 80)
    
    print("\nAdd --rsubt_control to enable risk-gated hyperparameter adjustment")
    print("Example:")
    print("  uv run python cleanrl/ppo_continuous_action.py --env_id Humanoid-v4 --seed 1 --rsubt_monitor --rsubt_control")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rsubt experiment benchmark commands"
    )
    parser.add_argument(
        "--suite", type=str, default="all",
        choices=["mujoco", "dmc", "atari", "all"],
        help="Benchmark suite to generate commands for"
    )
    parser.add_argument(
        "--algo", type=str, default="ppo",
        choices=["ppo", "sac", "all"],
        help="Algorithm to use"
    )
    parser.add_argument(
        "--stress", type=str, default=None,
        choices=["s1", "s3", None],
        help="Stress scenario to apply"
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for commands (optional)"
    )
    parser.add_argument(
        "--format", type=str, default="commands",
        choices=["commands", "benchmark", "summary"],
        help="Output format"
    )
    
    args = parser.parse_args()
    
    if args.suite == "all" or args.format == "summary":
        print_all_benchmarks()
        return
    
    seeds = list(range(1, args.seeds + 1))
    algos = ["ppo", "sac"] if args.algo == "all" else [args.algo]
    
    all_commands = []
    for algo in algos:
        if args.format == "benchmark":
            print(generate_benchmark_py_config(args.suite, algo, seeds, args.stress))
        else:
            cmds = generate_suite_commands(args.suite, algo, seeds, args.stress)
            all_commands.extend(cmds)
    
    if args.format == "commands":
        for cmd in all_commands:
            print(cmd)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated benchmark commands\n\n")
            for cmd in all_commands:
                f.write(cmd + "\n")
        print(f"\nCommands written to {args.output}")


if __name__ == "__main__":
    main()
