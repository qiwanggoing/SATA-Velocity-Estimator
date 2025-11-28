import os
from datetime import datetime
import numpy as np
import collections

# First, import Isaac Gym related modules
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR

# Then, import torch and other libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import VelocityEstimator


def collect_data(env, policy, num_steps, history_len=11, input_dim_per_step=33):
    """
    Collects data from the environment by running a given policy.
    Data is collected on the CPU to avoid GPU memory issues.
    This version manually manages the history buffer within the function.

    Args:
        env: The Isaac Gym environment.
        policy: The policy to generate actions.
        num_steps (int): The number of steps to collect data for.
        history_len (int): The length of the history sequence for the estimator.
        input_dim_per_step (int): The dimension of the input at each time step for the estimator.

    Returns:
        A tuple of (inputs, targets) for the estimator training as CPU tensors.
    """
    print(f"Collecting data for {num_steps} steps in {env.num_envs} environments...")
    
    # Buffers to store the collected data on the CPU
    all_inputs_list = []
    all_targets_list = []

    # Manually managed history buffers for each environment
    # Each deque stores (q, q_dot, at, wt, gt) for history_len steps
    history_deques = [collections.deque(maxlen=history_len) for _ in range(env.num_envs)]
    last_base_lin_vels = torch.zeros(env.num_envs, 3, device=env.device)

    obs = env.get_observations() # Initial observation

    # Warm-up phase to fill history buffers
    print("Warming up history buffers...")
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
        # Add Coriolis term to match deployment IMU physics (kinematic acceleration)
        coriolis_acc = torch.cross(current_base_ang_vel, current_base_lin_vel, dim=1)
        lin_acc = (current_base_lin_vel - last_base_lin_vels) / env.dt + coriolis_acc
        last_base_lin_vels = current_base_lin_vel.clone() # Update for next step's calculation

        # Construct current step's input for the estimator (q, q_dot, at, wt, gt)
        current_step_estimator_input = torch.cat((
            current_dof_pos,
            current_dof_vel,
            lin_acc,
            current_base_ang_vel,
            current_projected_gravity
        ), dim=-1).cpu() # Move to CPU immediately

        for env_idx in range(env.num_envs):
            history_deques[env_idx].append(current_step_estimator_input[env_idx])

    print("Starting data collection...")
    for i in range(num_steps):
        # --- Force Random Commands for Data Diversity ---
        if i % 100 == 0:
             # Randomize commands: [lin_vel_x, lin_vel_y, ang_vel_yaw]
             # Ranges: x [-1.5, 1.5], y [-1.0, 1.0], yaw [-1.5, 1.5]
             env.commands[:, 0] = torch.rand(env.num_envs, device=env.device) * 3.0 - 1.5
             env.commands[:, 1] = torch.rand(env.num_envs, device=env.device) * 2.0 - 1.0
             env.commands[:, 3] = torch.rand(env.num_envs, device=env.device) * 3.0 - 1.5 # Index 3 is often yaw in legged_gym, verify config. 
             # Note: standard legged_gym uses idx 0,1 for lin, 2 for yaw if heading_command=False. 
             # But many configs use 3 for yaw if heading=True. Assuming standard convention or checking env.
             # Let's assume standard 0,1,2 for now as it's safer for general "velocity" tracking.
             # However, if heading command is used, it might be different. 
             # Safer approach: Randomize both 2 and 3 to cover bases or check cfg.
             # Let's stick to indices 0, 1, and 2 (yaw velocity) as that's what the policy typically tracks.
             env.commands[:, 2] = torch.rand(env.num_envs, device=env.device) * 3.0 - 1.5

        with torch.no_grad():
            actions = policy(obs)
        
        # Step the environment
        obs, _, _, _, _ = env.step(actions)

        # Get current state for history update
        current_dof_pos = env.dof_pos.clone()
        current_dof_vel = env.dof_vel.clone()
        current_base_lin_vel = env.base_lin_vel.clone()
        current_base_ang_vel = env.base_ang_vel.clone()
        current_projected_gravity = env.projected_gravity.clone()

        # Calculate linear acceleration manually
        # Add Coriolis term to match deployment IMU physics (kinematic acceleration)
        coriolis_acc = torch.cross(current_base_ang_vel, current_base_lin_vel, dim=1)
        lin_acc = (current_base_lin_vel - last_base_lin_vels) / env.dt + coriolis_acc
        last_base_lin_vels = current_base_lin_vel.clone() # Update for next step's calculation

        # Construct current step's input for the estimator (q, q_dot, at, wt, gt)
        current_step_estimator_input = torch.cat((
            current_dof_pos,
            current_dof_vel,
            lin_acc,
            current_base_ang_vel,
            current_projected_gravity
        ), dim=-1).cpu() # Move to CPU immediately

        for env_idx in range(env.num_envs):
            history_deques[env_idx].append(current_step_estimator_input[env_idx])

            # Now that deque is full, we can form a sample
            estimator_input_sequence = torch.stack(list(history_deques[env_idx]), dim=0) # Shape: (history_len, input_dim_per_step)
            
            all_inputs_list.append(estimator_input_sequence)
            all_targets_list.append(current_base_lin_vel[env_idx].cpu()) # Target is current true linear velocity

        if i % 100 == 0:
            print(f"  ... collected step {i} / {num_steps}. Total samples: {len(all_inputs_list)}")

    print(f"Data collection complete. Total collected samples: {len(all_inputs_list)}. Concatenating tensors...")
    
    # Concatenate lists of tensors into single large tensors on the CPU
    if not all_inputs_list:
        raise ValueError("No data collected. Ensure num_steps is sufficient for history_len and policy moves.")
        
    inputs = torch.stack(all_inputs_list, dim=0) # Shape: (Total_samples, history_len, input_dim_per_step)
    targets = torch.stack(all_targets_list, dim=0) # Shape: (Total_samples, 3)
    
    return inputs, targets

def train(args):
    # --- Environment and Policy Setup ---
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Explicitly set num_envs to a reasonable value for training estimator to save GPU memory
    env_cfg.env.num_envs = 512 
    # Ensure history collection is NOT enabled in the environment itself
    # as we are managing it manually in collect_data
    # This line might not be strictly necessary if go2_torque_config.py is reverted,
    # but added for clarity and robustness.
    # The flags enable_estimator_history and use_velocity_estimator no longer exist in go2_torque_config.py
    # if hasattr(env_cfg.env, 'enable_estimator_history'):
    #     env_cfg.env.enable_estimator_history = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # --- Load a pre-trained policy to generate interesting data ---
    policy = ppo_runner.get_inference_policy(device=env.device)
    # --- Hardcoded policy path for training estimator ---
    # User confirmed this policy is valid and working.
    resume_path = "/home/qiwang/SATA/legged_gym/logs/SATA/Nov27_15-54-20_/model_3000.pt"
    print(f"Loading hardcoded policy from: {resume_path}")
    try:
        ppo_runner.load(resume_path)
    except Exception as e:
        print(f"Could not load hardcoded policy: {e}")
        print("Warning: Collecting data with a random (untrained) policy.")
    # The --load_run and --checkpoint arguments will be ignored for policy loading in this script.
    # if args.load_run:
    #     try:
    #         log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    #         resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    #         print(f"Loading policy from: {resume_path}")
    #         ppo_runner.load(resume_path)
    #     except Exception as e:
    #         print(f"Could not load policy: {e}")
    #         print("Warning: Collecting data with a random (untrained) policy.")

    # --- Data Collection ---
    num_collection_steps = 5000 # Increased to 5000 for better coverage
    # The history_len and input_dim_per_step parameters need to match the VelocityEstimator definition
    inputs, targets = collect_data(env, policy, num_collection_steps, history_len=11, input_dim_per_step=33)
    
    # --- Estimator Training ---
    print("Starting estimator training...")
    
    # Hyperparameters
    batch_size = 1024
    learning_rate = 1e-3
    num_epochs = 20
    
    # Create DataLoader from CPU tensors
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    estimator = VelocityEstimator(input_dim=inputs.shape[2]).to(env.device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(estimator.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Move current batch to GPU
            batch_inputs = batch_inputs.to(env.device)
            batch_targets = batch_targets.to(env.device)

            # Forward pass
            predictions = estimator(batch_inputs)
            loss = loss_fn(predictions, batch_targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}')
        
    print("Estimator training complete.")

    # --- Save the trained model ---
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'estimator')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'velocity_estimator_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt')
    torch.save(estimator.state_dict(), save_path)
    print(f"Trained velocity estimator saved to: {save_path}")


if __name__ == '__main__':
    # The default 'go2_torque' task is assumed for training the estimator
    # You can override this with --task <your_task>
    args = get_args(task_name='go2_torque') 
    train(args)
