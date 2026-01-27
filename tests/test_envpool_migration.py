"""
Tests for the EnvPool migration.

Tests include:
- Registry validity: All env IDs map correctly to valid EnvPool envs
- Script smoke tests: All new EnvPool scripts run without errors
- TensorBoard key contract: Required keys are logged correctly
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import pytest


# ============================================================================
# Registry Validity Tests
# ============================================================================

def test_envpool_registry_mujoco_ids_valid():
    """Test that all MuJoCo env IDs are valid in EnvPool."""
    import envpool
    from cleanrl_utils.envpool_registry import BENCHMARK_MUJOCO_IDS, validate_envpool_id
    
    for env_id in BENCHMARK_MUJOCO_IDS:
        is_valid, backend = validate_envpool_id(env_id)
        assert is_valid, f"MuJoCo env {env_id} is not valid"
        assert backend == "gym", f"MuJoCo env {env_id} should use gym backend"
        
        # Actually try to create the env
        env = envpool.make(env_id, env_type="gym", num_envs=1)
        assert env is not None
        env.close()


def test_envpool_registry_dmc_ids_valid():
    """Test that all DMC env IDs are valid in EnvPool."""
    import envpool
    from cleanrl_utils.envpool_registry import BENCHMARK_DMC_ENVPOOL_IDS, validate_envpool_id
    
    for env_id in BENCHMARK_DMC_ENVPOOL_IDS:
        is_valid, backend = validate_envpool_id(env_id)
        assert is_valid, f"DMC env {env_id} is not valid"
        assert backend == "dm", f"DMC env {env_id} should use dm backend"
        
        # Actually try to create the env
        env = envpool.make_dm(env_id, num_envs=1)
        assert env is not None
        env.close()


def test_envpool_registry_atari_ids_valid():
    """Test that all Atari env IDs are valid in EnvPool."""
    import envpool
    from cleanrl_utils.envpool_registry import BENCHMARK_ATARI_ENVPOOL_IDS, validate_envpool_id
    
    for env_id in BENCHMARK_ATARI_ENVPOOL_IDS:
        is_valid, backend = validate_envpool_id(env_id)
        assert is_valid, f"Atari env {env_id} is not valid"
        assert backend == "gym", f"Atari env {env_id} should use gym backend"
        
        # Actually try to create the env
        env = envpool.make(env_id, env_type="gym", num_envs=1)
        assert env is not None
        env.close()


def test_envpool_legacy_mapping_correctness():
    """Test that legacy ID mappings produce correct EnvPool IDs."""
    from cleanrl_utils.envpool_registry import (
        get_envpool_id,
        get_legacy_to_envpool_mapping,
    )
    
    mapping = get_legacy_to_envpool_mapping()
    
    for legacy_id, expected_envpool_id in mapping.items():
        actual_envpool_id = get_envpool_id(legacy_id)
        assert actual_envpool_id == expected_envpool_id, \
            f"Legacy {legacy_id} should map to {expected_envpool_id}, got {actual_envpool_id}"


def test_envpool_registry_self_check():
    """Test the registry's built-in self-check functions."""
    from cleanrl_utils.envpool_registry import (
        check_all_benchmark_ids_valid,
        check_legacy_mappings_valid,
    )
    
    assert check_all_benchmark_ids_valid(), "Some benchmark IDs are invalid"
    assert check_legacy_mappings_valid(), "Some legacy mappings are invalid"


# ============================================================================
# Script Smoke Tests
# ============================================================================

@pytest.mark.slow
def test_ppo_continuous_action_envpool_smoke():
    """Smoke test for PPO continuous action EnvPool script."""
    subprocess.run(
        "python cleanrl/ppo_continuous_action_envpool.py "
        "--env_id HalfCheetah-v4 "
        "--num_envs 2 "
        "--num_steps 64 "
        "--total_timesteps 128 "
        "--eval_interval_steps 0",
        shell=True,
        check=True,
    )


@pytest.mark.slow
def test_sac_continuous_action_envpool_smoke():
    """Smoke test for SAC continuous action EnvPool script."""
    subprocess.run(
        "python cleanrl/sac_continuous_action_envpool.py "
        "--env_id HalfCheetah-v4 "
        "--num_envs 1 "
        "--total_timesteps 500 "
        "--learning_starts 100 "
        "--eval_interval_steps 0",
        shell=True,
        check=True,
    )


@pytest.mark.slow
def test_td3_continuous_action_envpool_smoke():
    """Smoke test for TD3 continuous action EnvPool script."""
    subprocess.run(
        "python cleanrl/td3_continuous_action_envpool.py "
        "--env_id HalfCheetah-v4 "
        "--num_envs 1 "
        "--total_timesteps 500 "
        "--learning_starts 100 "
        "--eval_interval_steps 0",
        shell=True,
        check=True,
    )


@pytest.mark.slow
def test_sac_atari_envpool_smoke():
    """Smoke test for SAC Atari EnvPool script."""
    subprocess.run(
        "python cleanrl/sac_atari_envpool.py "
        "--env_id Breakout-v5 "
        "--num_envs 2 "
        "--total_timesteps 500 "
        "--learning_starts 100 "
        "--eval_interval_steps 0",
        shell=True,
        check=True,
    )


@pytest.mark.slow
def test_ppo_continuous_action_envpool_dmc_smoke():
    """Smoke test for PPO continuous action EnvPool script with DMC env."""
    subprocess.run(
        "python cleanrl/ppo_continuous_action_envpool.py "
        "--env_id CheetahRun-v1 "
        "--num_envs 2 "
        "--num_steps 64 "
        "--total_timesteps 128 "
        "--eval_interval_steps 0",
        shell=True,
        check=True,
    )


# ============================================================================
# TensorBoard Key Contract Tests
# ============================================================================

def _run_script_and_get_tb_keys(script_cmd: str, run_dir: str) -> set:
    """Run a script and extract all TensorBoard scalar keys from the event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    # Run the script
    subprocess.run(script_cmd, shell=True, check=True)
    
    # Find the event files
    runs_dir = Path(run_dir)
    event_files = list(runs_dir.rglob("events.out.tfevents.*"))
    
    if not event_files:
        # Try to find in runs subdirectory
        event_files = list(Path("runs").rglob("events.out.tfevents.*"))
    
    all_keys = set()
    for event_file in event_files:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        all_keys.update(ea.Tags().get("scalars", []))
    
    return all_keys


@pytest.mark.slow
def test_tb_contract_ppo_envpool_keys_present():
    """Test that PPO EnvPool script logs required TensorBoard keys."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        
        # Run PPO with eval enabled
        subprocess.run(
            "python cleanrl/ppo_continuous_action_envpool.py "
            "--env_id HalfCheetah-v4 "
            "--num_envs 2 "
            "--num_steps 64 "
            "--total_timesteps 256 "
            "--eval_interval_steps 100 "
            "--rsubt_monitor",
            shell=True,
            check=True,
            cwd="/home/ubuntu/ICMLActual/ICML2026",
        )
        
        # Find event files
        runs_dir = Path("/home/ubuntu/ICMLActual/ICML2026/runs")
        event_files = sorted(runs_dir.rglob("events.out.tfevents.*"), key=os.path.getmtime)
        
        if not event_files:
            pytest.skip("No TensorBoard event files found")
        
        # Get the most recent event file
        latest_event_file = event_files[-1]
        ea = EventAccumulator(str(latest_event_file.parent))
        ea.Reload()
        
        keys = set(ea.Tags().get("scalars", []))
        
        # Required keys for scoring suite
        required_keys = [
            "charts/episodic_return",  # Training returns
            "charts/learning_rate",
            "losses/value_loss",
            "losses/approx_kl",
            "losses/clipfrac",
            "losses/entropy",
            "losses/explained_variance",
            "baseline/grad_norm_actor",
            "baseline/grad_norm_critic",
        ]
        
        # Check required keys are present
        for key in required_keys:
            assert key in keys, f"Required TensorBoard key '{key}' not found. Found keys: {sorted(keys)}"
        
        # RSUBT keys should be present when rsubt_monitor is enabled
        rsubt_keys = [k for k in keys if k.startswith("rsubt/")]
        assert len(rsubt_keys) > 0, f"No RSUBT keys found when rsubt_monitor enabled. Found keys: {sorted(keys)}"
        
        # Clean up the run directory
        shutil.rmtree(latest_event_file.parent, ignore_errors=True)


@pytest.mark.slow  
def test_tb_contract_sac_envpool_keys_present():
    """Test that SAC EnvPool script logs required TensorBoard keys."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    # Run SAC
    subprocess.run(
        "python cleanrl/sac_continuous_action_envpool.py "
        "--env_id HalfCheetah-v4 "
        "--num_envs 1 "
        "--total_timesteps 1000 "
        "--learning_starts 200 "
        "--eval_interval_steps 500",
        shell=True,
        check=True,
        cwd="/home/ubuntu/ICMLActual/ICML2026",
    )
    
    # Find event files
    runs_dir = Path("/home/ubuntu/ICMLActual/ICML2026/runs")
    event_files = sorted(runs_dir.rglob("events.out.tfevents.*"), key=os.path.getmtime)
    
    if not event_files:
        pytest.skip("No TensorBoard event files found")
    
    # Get the most recent event file
    latest_event_file = event_files[-1]
    ea = EventAccumulator(str(latest_event_file.parent))
    ea.Reload()
    
    keys = set(ea.Tags().get("scalars", []))
    
    # Required keys for SAC
    required_keys = [
        "charts/episodic_return",
        "losses/actor_loss",
        "losses/qf1_loss",
        "losses/qf2_loss",
        "losses/alpha",
        "baseline/grad_norm_actor",
    ]
    
    for key in required_keys:
        assert key in keys, f"Required TensorBoard key '{key}' not found. Found keys: {sorted(keys)}"
    
    # eval/return_mean should be present when eval is enabled
    assert "eval/return_mean" in keys, f"eval/return_mean not found. Found keys: {sorted(keys)}"
    
    # Clean up
    shutil.rmtree(latest_event_file.parent, ignore_errors=True)


# ============================================================================
# Integration Tests
# ============================================================================

def test_envpool_make_env_helper():
    """Test the make_envpool_env helper function."""
    from cleanrl_utils.envpool_registry import make_envpool_env
    
    # Test MuJoCo with legacy ID
    env = make_envpool_env("HalfCheetah-v4", num_envs=2)
    assert env is not None
    obs = env.reset()
    assert obs.shape[0] == 2  # num_envs
    env.close()
    
    # Test DMC with legacy ID
    env = make_envpool_env("dm_control/cheetah-run-v0", num_envs=2)
    assert env is not None
    env.close()
    
    # Test Atari with legacy ID
    env = make_envpool_env("BreakoutNoFrameskip-v4", num_envs=2)
    assert env is not None
    obs = env.reset()
    assert obs.shape[0] == 2
    env.close()
    
    # Test with native EnvPool ID
    env = make_envpool_env("Breakout-v5", num_envs=2)
    assert env is not None
    env.close()


def test_benchmark_recipes_envpool_generation():
    """Test that benchmark recipes generate valid EnvPool commands."""
    from cleanrl_utils.rsubt.benchmark_recipes import generate_full_benchmark_script
    
    script = generate_full_benchmark_script(
        num_seeds=2,
        workers=1,
        use_envpool=True,
        num_envs=4,
    )
    
    # Check that EnvPool scripts are used
    assert "ppo_continuous_action_envpool.py" in script
    assert "sac_continuous_action_envpool.py" in script
    assert "td3_continuous_action_envpool.py" in script
    
    # Check that num_envs is passed
    assert "--num_envs 4" in script
    
    # Check that the script is valid bash
    assert script.startswith("#!/bin/bash")


def test_envpool_dmc_flatten_observation_shape():
    """DMC observations should flatten to (num_envs, obs_dim) for EnvPool scripts."""
    import envpool
    from cleanrl_utils.envpool_vecenv import flatten_dmc_observation

    num_envs = 3
    env = envpool.make_dm("CheetahRun-v1", num_envs=num_envs, seed=1)
    ts = env.reset()
    flat = flatten_dmc_observation(ts.observation, num_envs)
    assert flat.shape[0] == num_envs
    assert flat.ndim == 2
    env.close()


def test_envpool_gym_truncation_split():
    """Gym-style EnvPool env should expose TimeLimit.truncated and split correctly."""
    import envpool
    import numpy as np
    from cleanrl_utils.envpool_vecenv import split_terminations_truncations

    env = envpool.make("HalfCheetah-v4", env_type="gym", num_envs=1, seed=1)
    obs = env.reset()
    a = np.zeros((1,) + env.action_space.shape, dtype=np.float32)
    for _ in range(1500):
        next_obs, rew, done, info = env.step(a)
        if bool(done[0]):
            split = split_terminations_truncations(done, info)
            # HalfCheetah ends at time limit: should be truncation, not termination
            assert bool(split.truncations[0]) is True
            assert bool(split.terminations[0]) is False
            break
    else:
        raise AssertionError("Env did not reach done within expected steps")
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
