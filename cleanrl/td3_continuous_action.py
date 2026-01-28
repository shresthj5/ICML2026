# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Rsubt stability certificate arguments
    rsubt_monitor: bool = False
    """if toggled, enables Rsubt stability certificate monitoring"""
    rsubt_control: bool = False
    """if toggled, enables risk-gated hyperparameter control based on Rsubt"""
    rsubt_sketch_dim: int = 8192
    """dimension of the sketch space for Rsubt"""
    rsubt_buffer_size: int = 64
    """number of updates to store in Rsubt buffer"""
    rsubt_k_max: int = 20
    """maximum number of eigenvalues to track"""
    rsubt_diag_interval: int = 10
    """DEPRECATED: compute Rsubt diagnostics every N actor updates (use diag_interval_steps instead)"""
    rsubt_diag_interval_steps: int = 20480
    """compute Rsubt diagnostics every N environment steps (step-based, consistent across num_envs)"""
    rsubt_min_buffer_entries: int = 4
    """minimum buffer entries before computing Rsubt (controls warmup speed)"""
    rsubt_tau_yellow: float = 0.5
    """Rsubt EWMA threshold for YELLOW state (calibrate or override)"""
    rsubt_tau_red: float = 1.0
    """Rsubt EWMA threshold for RED state (calibrate or override)"""
    rsubt_thresholds: str = ""
    """optional path to thresholds JSON from `python -m cleanrl_utils.rsubt.calibrate_thresholds` (overrides tau_yellow/tau_red)"""

    # Periodic evaluation arguments
    eval_interval_steps: int = 10000
    """evaluate every N environment steps (0 to disable)"""
    eval_episodes: int = 10
    """number of episodes per evaluation"""
    eval_seed: int = 12345
    """fixed seed for evaluation reproducibility"""
    eval_deterministic: bool = True
    """use deterministic actions for evaluation"""

    # Stress test arguments
    stress_preset: str = ""
    """stress test preset name (see cleanrl_utils.stress_wrappers.STRESS_PRESETS)"""
    obs_noise_std: float = 0.0
    """observation noise std (0 to disable)"""
    obs_dropout_p: float = 0.0
    """observation dropout probability (0 to disable)"""
    action_delay: int = 0
    """action delay in steps (0 to disable)"""
    reward_scale: float = 1.0
    """reward scaling factor"""


def make_env(
    env_id, seed, idx, capture_video, run_name,
    stress_preset: str = "",
    obs_noise_std: float = 0.0,
    obs_dropout_p: float = 0.0,
    action_delay: int = 0,
    reward_scale: float = 1.0,
):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        
        # Apply stress wrappers
        if stress_preset or obs_noise_std > 0 or obs_dropout_p > 0 or action_delay > 0 or reward_scale != 1.0:
            from cleanrl_utils.stress_wrappers import (
                apply_stress_wrappers,
                ObsNoiseWrapper,
                ObsDropoutWrapper,
                ActionDelayWrapper,
                RewardScaleWrapper,
            )
            
            if stress_preset:
                env = apply_stress_wrappers(env, preset=stress_preset, seed=seed + idx)
            if obs_noise_std > 0:
                env = ObsNoiseWrapper(env, std=obs_noise_std, seed=seed + idx)
            if obs_dropout_p > 0:
                env = ObsDropoutWrapper(env, p=obs_dropout_p, seed=seed + idx)
            if action_delay > 0:
                env = ActionDelayWrapper(env, delay_steps=action_delay)
            if reward_scale != 1.0:
                env = RewardScaleWrapper(env, scale=reward_scale)
        
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def get_hidden(self, x):
        """Get hidden layer activation for rank computation."""
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return h2


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(
            args.env_id, args.seed + i, i, args.capture_video, run_name,
            stress_preset=args.stress_preset,
            obs_noise_std=args.obs_noise_std,
            obs_dropout_p=args.obs_dropout_p,
            action_delay=args.action_delay,
            reward_scale=args.reward_scale,
        ) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # Rsubt stability certificate setup
    rsubt_monitor = None
    rsubt_actor_update_count = 0
    if args.rsubt_monitor:
        from cleanrl_utils.rsubt import RsubtMonitor
        rsubt_monitor = RsubtMonitor(
            params=list(actor.parameters()),
            sketch_dim=args.rsubt_sketch_dim,
            buffer_size=args.rsubt_buffer_size,
            k_max=args.rsubt_k_max,
            diag_interval=args.rsubt_diag_interval,
            diag_interval_steps=args.rsubt_diag_interval_steps,
            min_buffer_entries=args.rsubt_min_buffer_entries,
            algorithm="td3",
            device=device,
            seed=args.seed,
            enable_controller=args.rsubt_control,
            thresholds_path=args.rsubt_thresholds if args.rsubt_thresholds else None,
            tau_yellow=None if args.rsubt_thresholds else args.rsubt_tau_yellow,
            tau_red=None if args.rsubt_thresholds else args.rsubt_tau_red,
            base_lr=args.learning_rate,
            base_policy_freq=args.policy_frequency,
            base_exploration_noise=args.exploration_noise,
        )

    # Periodic evaluation setup
    online_evaluator = None
    last_eval_step = 0
    if args.eval_interval_steps > 0:
        from cleanrl_utils.evals.online_eval import OnlineEvaluator
        online_evaluator = OnlineEvaluator(
            make_env_fn=make_env,
            env_id=args.env_id,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            device=device,
            deterministic=args.eval_deterministic,
            gamma=args.gamma,
            capture_video=False,
            run_name=f"{run_name}-eval",
        )

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Track current exploration noise (can be adjusted by Rsubt controller)
    current_exploration_noise = args.exploration_noise

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * current_exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                # Rsubt: check if we should skip actor updates
                skip_actor = False
                if rsubt_monitor is not None and args.rsubt_control:
                    hp = rsubt_monitor.get_hyperparams()
                    if hp is not None:
                        if hp.skip_actor:
                            skip_actor = True
                        actor_optimizer.param_groups[0]["lr"] = hp.actor_lr
                        current_exploration_noise = hp.exploration_noise

                if not skip_actor:
                    # Rsubt: snapshot actor params before update
                    if rsubt_monitor is not None:
                        rsubt_monitor.snapshot_params(list(actor.parameters()))

                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    
                    # Compute gradient norm before step
                    grad_norm_actor = sum(p.grad.norm().item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
                    
                    actor_optimizer.step()

                    # Rsubt: record update and compute certificate
                    if rsubt_monitor is not None:
                        rsubt_actor_update_count += 1
                        rsubt_monitor.update(list(actor.parameters()), global_step)
                        rsubt_metrics = rsubt_monitor.maybe_compute(global_step)
                        if rsubt_metrics is not None:
                            rsubt_monitor.log_to_tensorboard(writer, global_step)

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                
                # Baseline diagnostics: gradient norms
                if 'grad_norm_actor' in dir():
                    writer.add_scalar("baseline/grad_norm_actor", grad_norm_actor, global_step)
                
                # Representation rank diagnostics
                if args.rsubt_monitor and global_step % 1000 == 0:
                    from cleanrl_utils.rsubt.reprank import compute_effective_rank, compute_stable_rank
                    with torch.no_grad():
                        actor_hidden = actor.get_hidden(data.observations)
                        actor_effrank = compute_effective_rank(actor_hidden)
                        actor_stablerank = compute_stable_rank(actor_hidden)
                    writer.add_scalar("reprank/actor_effrank", actor_effrank, global_step)
                    writer.add_scalar("reprank/actor_stablerank", actor_stablerank, global_step)
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

        # Periodic evaluation
        if online_evaluator is not None and (global_step - last_eval_step) >= args.eval_interval_steps:
            eval_stats = online_evaluator.evaluate_offpolicy(actor, exploration_noise=0.0)
            online_evaluator.log_to_tensorboard(writer, global_step, eval_stats)
            print(f"eval: global_step={global_step}, return_mean={eval_stats.return_mean:.2f}, return_std={eval_stats.return_std:.2f}")
            last_eval_step = global_step

    # Close evaluator
    if online_evaluator is not None:
        online_evaluator.close()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "TD3",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
