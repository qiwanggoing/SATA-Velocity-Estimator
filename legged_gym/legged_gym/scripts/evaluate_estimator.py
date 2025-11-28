import os
import numpy as np
import collections

# First, import Isaac Gym related modules
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR

# Then, import torch and other libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import VelocityEstimator

def evaluate(args, history_len=11, input_dim_per_step=33):
    # --- Environment and Policy Setup ---
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Explicitly set num_envs to 1 for evaluation (as play.py usually does)
    env_cfg.env.num_envs = 1 
    # Remove reference to enable_estimator_history as it's no longer in config
    # if hasattr(env_cfg.env, 'enable_estimator_history'):
    #     env_cfg.env.enable_estimator_history = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # --- Load a pre-trained policy to generate movement ---
    policy = ppo_runner.get_inference_policy(device=env.device)
    # --- Hardcoded policy path for evaluation ---
    # User confirmed this policy is valid and working.
    resume_path = "/home/qiwang/SATA/legged_gym/logs/SATA/Oct27_13-13-44_/model_3000.pt"
    print(f"Loading hardcoded policy from: {resume_path}")
    try:
        ppo_runner.load(resume_path)
    except Exception as e:
        print(f"Could not load hardcoded policy: {e}")
        print("Warning: Evaluating with a random (untrained) policy.")
    # The --load_run and --checkpoint arguments will be ignored for policy loading in this script.
    # if args.load_run:
    #     try:
    #         log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    #         resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    #         print(f"Loading policy from: {resume_path}")
    #         ppo_runner.load(resume_path)
    #     except Exception as e:
    #         print(f"Could not load policy: {e}")
    #         print("Warning: Evaluating with a random (untrained) policy.")
            
    # --- Load the trained estimator ---
    estimator = VelocityEstimator(input_dim=input_dim_per_step).to(env.device)
    if not args.estimator_path:
        raise ValueError("Estimator path must be provided with --estimator_path")
    
    estimator_path = os.path.join(LEGGED_GYM_ROOT_DIR, args.estimator_path)
    print(f"Loading estimator from: {estimator_path}")
    try:
        estimator.load_state_dict(torch.load(estimator_path))
        estimator.eval() # Set to evaluation mode
    except Exception as e:
        raise FileNotFoundError(f"Failed to load estimator model from {estimator_path}. Error: {e}")

    # --- Evaluation Loop ---
    num_eval_steps = 1000
    print(f"Evaluating for {num_eval_steps} steps...")
    
    # Manually managed history buffer for the single evaluation environment
    history_deque = collections.deque(maxlen=history_len)
    last_base_lin_vel = torch.zeros(1, 3, device=env.device) # For acceleration calculation

    true_velocities = torch.zeros(num_eval_steps, 3)
    predicted_velocities = torch.zeros(num_eval_steps, 3)

    obs = env.get_observations() # Initial observation

    # Warm-up phase to fill history buffer
    print("Warming up history buffer...")
    for _ in range(history_len - 1): # Fill (history_len - 1) steps
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _, _ = env.step(actions)

        # Get current state for history update
        current_dof_pos = env.dof_pos.clone()
        current_dof_vel = env.dof_vel.clone()
        current_base_lin_vel = env.base_lin_vel.clone()
        current_base_ang_vel = env.base_ang_vel.clone()
        current_projected_gravity = env.projected_gravity.clone()

        # Calculate linear acceleration manually
        lin_acc = (current_base_lin_vel - last_base_lin_vel) / env.dt
        last_base_lin_vel = current_base_lin_vel.clone()

        current_step_estimator_input = torch.cat((
            current_dof_pos,
            current_dof_vel,
            lin_acc,
            current_base_ang_vel,
            current_projected_gravity
        ), dim=-1).cpu().squeeze(0) # Squeeze because num_envs=1

        history_deque.append(current_step_estimator_input)

    print("Starting evaluation data collection...")
    for i in range(num_eval_steps):
        with torch.no_grad():
            # Policy action
            actions = policy(obs)
            
            # Get current state for history update (before stepping env for next obs)
            current_dof_pos = env.dof_pos.clone()
            current_dof_vel = env.dof_vel.clone()
            current_base_lin_vel = env.base_lin_vel.clone()
            current_base_ang_vel = env.base_ang_vel.clone()
            current_projected_gravity = env.projected_gravity.clone()

            # Calculate linear acceleration manually
            lin_acc = (current_base_lin_vel - last_base_lin_vel) / env.dt
            last_base_lin_vel = current_base_lin_vel.clone()

            current_step_estimator_input = torch.cat((
                current_dof_pos,
                current_dof_vel,
                lin_acc,
                current_base_ang_vel,
                current_projected_gravity
            ), dim=-1).cpu().squeeze(0)
            
            history_deque.append(current_step_estimator_input)
            estimator_input_sequence = torch.stack(list(history_deque), dim=0).unsqueeze(0).to(env.device) # Add batch dim and move to GPU
            
            # Estimator prediction
            predicted_vel = estimator(estimator_input_sequence).squeeze(0) # Remove batch dim

        obs, _, _, _, _ = env.step(actions) # Step the environment after collecting data for current time step
        
        # Store results for the selected environment (env_idx=0 as num_envs=1)
        true_velocities[i] = current_base_lin_vel.squeeze(0)
        predicted_velocities[i] = predicted_vel.cpu()

    # --- Calculate Metrics ---
    loss_fn = nn.MSELoss()
    mse = loss_fn(predicted_velocities, true_velocities)
    print(f"\nEvaluation Complete.")
    print(f"Mean Squared Error (MSE) on {num_eval_steps} steps: {mse.item():.6f}")

    # --- Visualization ---
    print("Generating plot...")
    time_axis = np.arange(num_eval_steps) * env.dt

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Velocity Estimation: Predicted vs. Ground Truth', fontsize=16)

    # Plot X velocity
    axs[0].plot(time_axis, true_velocities[:, 0].cpu().numpy(), label='Ground Truth', color='blue')
    axs[0].plot(time_axis, predicted_velocities[:, 0].cpu().numpy(), label='Predicted', color='red', linestyle='--')
    axs[0].set_ylabel('Velocity X (m/s)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Y velocity
    axs[1].plot(time_axis, true_velocities[:, 1].cpu().numpy(), label='Ground Truth', color='blue')
    axs[1].plot(time_axis, predicted_velocities[:, 1].cpu().numpy(), label='Predicted', color='red', linestyle='--')
    axs[1].set_ylabel('Velocity Y (m/s)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Z velocity
    axs[2].plot(time_axis, true_velocities[:, 2].cpu().numpy(), label='Ground Truth', color='blue')
    axs[2].plot(time_axis, predicted_velocities[:, 2].cpu().numpy(), label='Predicted', color='red', linestyle='--')
    axs[2].set_ylabel('Velocity Z (m/s)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid(True)

    # Save the plot
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'estimator')
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'velocity_estimation_comparison.png')
    plt.savefig(plot_path)
    
    print(f"Plot saved to: {plot_path}")
    # To display the plot in an interactive session, you might use plt.show()
    # plt.show()

if __name__ == '__main__':
    # Add custom arguments for the evaluator
    estimator_custom_parameters = [
        {"name": "--estimator_path", "type": str, "required": True, "help": "Path to the trained velocity estimator model, relative to the LEGGED_GYM_ROOT_DIR."}
    ]
    args = get_args(task_name='go2_torque', custom_parameters=estimator_custom_parameters)
    evaluate(args)
