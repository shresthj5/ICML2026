#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/ICMLActual/ICML2026
export MUJOCO_GL=osmesa

WORKERS=2
SEEDS=$(seq 1 10)

# Timesteps per domain
MUJOCO_STEPS=10000000
DMC_STEPS=5000000
ATARI_STEPS=30000000

# Parallel envs - 64 for better utilization
NUM_ENVS_PPO=64
NUM_ENVS_ATARI=64
NUM_ENVS_OFFPOLICY=2

# Adjusted num_steps to keep batch_size identical
NUM_STEPS_PPO=1024      # was 2048, halved to maintain batch_size=65536
NUM_STEPS_ATARI=64      # was 128, halved to maintain batch_size=4096

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

for seed in ${SEEDS}; do
  echo "=============================="
  echo "SEED ${seed}"
  echo "=============================="

  # ===================================
  # SKIP FOR SEED 1 ONLY (already completed)
  # ===================================
  if [ ${seed} -ge 2 ]; then
    # ----- PPO MuJoCo A0 -----
    run_benchmark "${MUJOCO_ENVS}" \
      "python cleanrl/ppo_continuous_action_envpool.py --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4" \
      ${seed}

    # ----- PPO MuJoCo A1 -----
    run_benchmark "${MUJOCO_ENVS}" \
      "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4" \
      ${seed}

    # ----- PPO DMC A0 -----
    run_benchmark "${DMC_ENVS}" \
      "python cleanrl/ppo_continuous_action_envpool.py --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS} --no-anneal-lr --learning-rate 3e-4" \
      ${seed}
  fi

  # ===================================
  # RESUME POINT FOR SEED 1: PPO DMC A1
  # ===================================

  # ----- PPO DMC A1 -----
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS} --no-anneal-lr --learning-rate 3e-4" \
    ${seed}

  # ----- PPO Atari A0 -----
  run_benchmark "${ATARI_ENVS}" \
    "python cleanrl/ppo_atari_envpool.py --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS} --no-anneal-lr" \
    ${seed}

  # ----- PPO Atari A1 -----
  run_benchmark "${ATARI_ENVS}" \
    "python cleanrl/ppo_atari_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS} --no-anneal-lr" \
    ${seed}

  # ----- SAC MuJoCo A0 -----
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS}" \
    ${seed}

  # ----- SAC MuJoCo A1 -----
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS}" \
    ${seed}

  # ----- SAC DMC A0 -----
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS}" \
    ${seed}

  # ----- SAC DMC A1 -----
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS}" \
    ${seed}

  # ----- SAC Atari A0 -----
  run_benchmark "${ATARI_ENVS}" \
    "python cleanrl/sac_atari_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${ATARI_STEPS}" \
    ${seed}

  # ----- SAC Atari A1 -----
  run_benchmark "${ATARI_ENVS}" \
    "python cleanrl/sac_atari_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${ATARI_STEPS}" \
    ${seed}

  # ----- TD3 MuJoCo A0 -----
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS}" \
    ${seed}

  # ----- TD3 MuJoCo A1 -----
  run_benchmark "${MUJOCO_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS}" \
    ${seed}

  # ----- TD3 DMC A0 -----
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS}" \
    ${seed}

  # ----- TD3 DMC A1 -----
  run_benchmark "${DMC_ENVS}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS}" \
    ${seed}

  # ===================================
  # STRESS TESTS
  # ===================================

  # ----- PPO Stress: Value Domination (MuJoCo) A1 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}

  # ----- PPO Stress: Value Domination (MuJoCo) A2 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}

  # ----- PPO Stress: Value Domination (DMC) A1 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}

  # ----- PPO Stress: Value Domination (DMC) A2 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS} --no-anneal-lr --learning-rate 3e-4 --vf-coef 4.0 --clip-coef 0.05 --update-epochs 15" \
    ${seed}

  # ----- PPO Stress: Observation Noise (MuJoCo) A1 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}

  # ----- PPO Stress: Observation Noise (MuJoCo) A2 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}

  # ----- PPO Stress: Observation Noise (DMC) A1 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}

  # ----- PPO Stress: Observation Noise (DMC) A2 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${DMC_STEPS} --no-anneal-lr --learning-rate 3e-4 --obs-noise-std 0.1" \
    ${seed}

  # ----- PPO Stress: Action Delay (MuJoCo) A1 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4 --action-delay 3" \
    ${seed}

  # ----- PPO Stress: Action Delay (MuJoCo) A2 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/ppo_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_PPO} --num-steps ${NUM_STEPS_PPO} --total-timesteps ${MUJOCO_STEPS} --no-anneal-lr --learning-rate 3e-4 --action-delay 3" \
    ${seed}

  # ----- SAC Stress: High LR (MuJoCo) A1 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}

  # ----- SAC Stress: High LR (MuJoCo) A2 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}

  # ----- SAC Stress: High LR (DMC) A1 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}

  # ----- SAC Stress: High LR (DMC) A2 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/sac_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS} --policy-lr 1e-3 --q-lr 3e-3" \
    ${seed}

  # ----- TD3 Stress: High LR (MuJoCo) A1 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS} --learning-rate 1e-3" \
    ${seed}

  # ----- TD3 Stress: High LR (MuJoCo) A2 -----
  run_benchmark "${FRAGILE_MUJOCO}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${MUJOCO_STEPS} --learning-rate 1e-3" \
    ${seed}

  # ----- TD3 Stress: High LR (DMC) A1 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS} --learning-rate 1e-3" \
    ${seed}

  # ----- TD3 Stress: High LR (DMC) A2 -----
  run_benchmark "${FRAGILE_DMC}" \
    "python cleanrl/td3_continuous_action_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${DMC_STEPS} --learning-rate 1e-3" \
    ${seed}

  # ----- PPO Stress: Value Domination (Atari) A1 -----
  run_benchmark "${FRAGILE_ATARI}" \
    "python cleanrl/ppo_atari_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS} --no-anneal-lr --vf-coef 2.0 --clip-coef 0.05" \
    ${seed}

  # ----- PPO Stress: Value Domination (Atari) A2 -----
  run_benchmark "${FRAGILE_ATARI}" \
    "python cleanrl/ppo_atari_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_ATARI} --num-steps ${NUM_STEPS_ATARI} --total-timesteps ${ATARI_STEPS} --no-anneal-lr --vf-coef 2.0 --clip-coef 0.05" \
    ${seed}

  # ----- SAC Stress: High LR (Atari) A1 -----
  run_benchmark "${FRAGILE_ATARI}" \
    "python cleanrl/sac_atari_envpool.py --rsubt-monitor --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${ATARI_STEPS} --policy-lr 1e-3 --q-lr 1e-3" \
    ${seed}

  # ----- SAC Stress: High LR (Atari) A2 -----
  run_benchmark "${FRAGILE_ATARI}" \
    "python cleanrl/sac_atari_envpool.py --rsubt-monitor --rsubt-control --num-envs ${NUM_ENVS_OFFPOLICY} --total-timesteps ${ATARI_STEPS} --policy-lr 1e-3 --q-lr 1e-3" \
    ${seed}

done

echo "=== FULL ICML SUITE COMPLETE ==="
