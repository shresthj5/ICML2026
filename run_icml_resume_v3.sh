#!/usr/bin/env bash
# Resume script v3 - Fixed SAC/TD3 to use standard config (num_envs=1)
# Current state: PPO MuJoCo/DMC A0/A1 done for seed 1, SAC needs restart with correct config
set -euo pipefail

cd /home/ubuntu/ICMLActual/ICML2026
export MUJOCO_GL=osmesa

WORKERS=2

# Timesteps per domain - PPO uses iterations (with large batch), SAC/TD3 use env steps
MUJOCO_STEPS_PPO=10000000
DMC_STEPS_PPO=5000000
ATARI_STEPS_PPO=30000000

# SAC/TD3: Standard 1M for MuJoCo, 500K for DMC (matches literature)
MUJOCO_STEPS_OFFPOLICY=1000000
DMC_STEPS_OFFPOLICY=500000

# Parallel envs - PPO benefits from many, SAC/TD3 use standard num_envs=1
NUM_ENVS_PPO=64
NUM_ENVS_ATARI=64
NUM_ENVS_OFFPOLICY=1  # Standard SAC/TD3 config (UTD=1:1)

# Num steps per rollout (PPO only)
NUM_STEPS_PPO=1024
NUM_STEPS_ATARI=64

# EnvPool environment IDs
MUJOCO_ENVS="HalfCheetah-v4 Hopper-v4 Walker2d-v4 Humanoid-v4"
DMC_ENVS="CheetahRun-v1 WalkerWalk-v1 WalkerRun-v1 HumanoidRun-v1"
ATARI_ENVS="Pong-v5 Breakout-v5 Seaquest-v5"

# Fragile subsets for stress testing
FRAGILE_MUJOCO="Humanoid-v4 Hopper-v4"
FRAGILE_DMC="HumanoidRun-v1 WalkerRun-v1"
FRAGILE_ATARI="Seaquest-v5"

run_benchmark() {
  local envs="$1"
  local cmd="$2"
  local seed="$3"
  uv run python -m cleanrl_utils.benchmark \
    --env-ids ${envs} \
    --command "${cmd}" \
    --num-seeds 1 \
    --workers ${WORKERS} \
    --start-seed ${seed}
}

run_mujoco_dmc_baselines() {
  local seed=$1
  
  # PPO MuJoCo A0, A1
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/ppo_continuous_action_envpool.py --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4" \
    ${seed}
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4" \
    ${seed}

  # PPO DMC A0, A1
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/ppo_continuous_action_envpool.py --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4" \
    ${seed}
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4" \
    ${seed}

  # SAC MuJoCo A0, A1 (standard config: num_envs=1, 1M steps)
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
    ${seed}
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
    ${seed}

  # SAC DMC A0, A1 (standard config: num_envs=1, 500K steps)
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
    ${seed}
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
    ${seed}

  # TD3 MuJoCo A0, A1 (standard config: num_envs=1, 1M steps)
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
    ${seed}
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
    ${seed}

  # TD3 DMC A0, A1 (standard config: num_envs=1, 500K steps)
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
    ${seed}
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
    ${seed}
}

run_mujoco_dmc_stress() {
  local seed=$1

  # PPO Stress: Value Domination (MuJoCo) A1, A2
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}

  # PPO Stress: Value Domination (DMC) A1, A2
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}

  # PPO Stress: Observation Noise (MuJoCo) A1, A2
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}

  # PPO Stress: Observation Noise (DMC) A1, A2
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}

  # PPO Stress: Action Delay (MuJoCo) A1, A2
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --action-delay 3" \
    ${seed}
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS_PPO} --no-anneal-lr --learning-rate 3e-4 --action-delay 3" \
    ${seed}

  # SAC Stress: High LR (MuJoCo) A1, A2
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}

  # SAC Stress: High LR (DMC) A1, A2
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}

  # TD3 Stress: High LR (MuJoCo) A1, A2
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY} --learning-rate 1e-3" \
    ${seed}
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY} --learning-rate 1e-3" \
    ${seed}

  # TD3 Stress: High LR (DMC) A1, A2
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY} --learning-rate 1e-3" \
    ${seed}
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY} --learning-rate 1e-3" \
    ${seed}
}

run_atari_baselines() {
  local seed=$1

  # PPO Atari A0, A1
  run_benchmark "${ATARI_ENVS}" \
    "python cleanrl/ppo_atari_envpool.py --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS_PPO} --no-anneal-lr" \
    ${seed}
  run_benchmark "${ATARI_ENVS}" \
    "python cleanrl/ppo_atari_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS_PPO} --no-anneal-lr" \
    ${seed}
}

run_atari_stress() {
  local seed=$1

  # PPO Stress: Value Domination (Atari) A1, A2
  run_benchmark "${FRAGILE_ATARI}" \
    "python cleanrl/ppo_atari_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS_PPO} --no-anneal-lr --vf-coef 2.0 --clip-coef 0.05" \
    ${seed}
  run_benchmark "${FRAGILE_ATARI}" \
    "python cleanrl/ppo_atari_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS_PPO} --no-anneal-lr --vf-coef 2.0 --clip-coef 0.05" \
    ${seed}
}

# ============================================================
# RESUME POINT v3: SAC/TD3 fixed to standard config
# Completed: PPO MuJoCo/DMC A0/A1 seed 1
# Restart: SAC from beginning with correct config
# ============================================================

echo "=============================================="
echo "RESUME v3: Fixed SAC/TD3 config (num_envs=1)"
echo "SAC: 1M MuJoCo, 500K DMC (~15-20 min each)"
echo "TD3: 1M MuJoCo, 500K DMC (~15-20 min each)"
echo "=============================================="

# Seed 1: SAC baselines A0, A1 (restart with correct config)
echo "=== Seed 1: SAC MuJoCo A0 ==="
run_benchmark "${MUJOCO_ENVS}" \
  "python cleanrl/sac_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
  1

echo "=== Seed 1: SAC MuJoCo A1 ==="
run_benchmark "${MUJOCO_ENVS}" \
  "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
  1

echo "=== Seed 1: SAC DMC A0 ==="
run_benchmark "${DMC_ENVS}" \
  "python cleanrl/sac_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
  1

echo "=== Seed 1: SAC DMC A1 ==="
run_benchmark "${DMC_ENVS}" \
  "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
  1

# Seed 1: TD3 baselines A0, A1
echo "=== Seed 1: TD3 MuJoCo A0 ==="
run_benchmark "${MUJOCO_ENVS}" \
  "python cleanrl/td3_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
  1

echo "=== Seed 1: TD3 MuJoCo A1 ==="
run_benchmark "${MUJOCO_ENVS}" \
  "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS_OFFPOLICY}" \
  1

echo "=== Seed 1: TD3 DMC A0 ==="
run_benchmark "${DMC_ENVS}" \
  "python cleanrl/td3_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
  1

echo "=== Seed 1: TD3 DMC A1 ==="
run_benchmark "${DMC_ENVS}" \
  "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS_OFFPOLICY}" \
  1

# Seed 1: Stress tests
echo "=== Seed 1: Stress Tests ==="
run_mujoco_dmc_stress 1

echo "=============================================="
echo "PHASE 1: Seeds 2-5 - MuJoCo + DMC ONLY"
echo "=============================================="

for seed in 2 3 4 5; do
  echo "=============================="
  echo "SEED ${seed} (Phase 1)"
  echo "=============================="
  run_mujoco_dmc_baselines ${seed}
  run_mujoco_dmc_stress ${seed}
done

# ============================================================
# PHASE 2: Seeds 6-10 - MuJoCo + DMC + Atari
# ============================================================
echo "=============================================="
echo "PHASE 2: Seeds 6-10 - MuJoCo + DMC + Atari"
echo "=============================================="

for seed in 6 7 8 9 10; do
  echo "=============================="
  echo "SEED ${seed} (Phase 2)"
  echo "=============================="
  run_mujoco_dmc_baselines ${seed}
  run_mujoco_dmc_stress ${seed}
  run_atari_baselines ${seed}
  run_atari_stress ${seed}
done

# ============================================================
# PHASE 3: Seeds 1-5 - Atari ONLY (complete 10 seeds for Atari)
# ============================================================
echo "=============================================="
echo "PHASE 3: Seeds 1-5 - Atari ONLY"
echo "=============================================="

for seed in 1 2 3 4 5; do
  echo "=============================="
  echo "SEED ${seed} (Phase 3: Atari)"
  echo "=============================="
  run_atari_baselines ${seed}
  run_atari_stress ${seed}
done

echo "=== FULL ICML SUITE COMPLETE ==="
