# Velocity Estimator Implementation Summary

This document summarizes the implementation details, training pipeline, and deployment architecture for the LSTM-based Velocity Estimator used in the Sim2Sim deployment of the RL policy on the Unitree Go2 robot.

## 1. Overview

The velocity estimator is a supervised learning model (LSTM) designed to estimate the robot's base linear velocity ($v_x, v_y, v_z$) using only proprioceptive sensors (IMU and Joint Encoders). This is critical for real-world deployment where ground truth velocity is unavailable.

## 2. Model Architecture

*   **Type**: LSTM (Long Short-Term Memory)
*   **Input Dimension**: 33
*   **Hidden Dimension**: 128
*   **Output Dimension**: 3 (Linear Velocity: $v_x, v_y, v_z$)
*   **Sequence Length (History)**: 11 steps (approx. 0.22s at 50Hz control loop)

## 3. Input Features (Observation Space)

The estimator takes a sequence of 11 time steps. At each step, the input is a **33-dimensional vector** composed of the following features, concatenated in this exact order:

1.  **Joint Positions ($q$)**: 12 dims
    *   Order: FL, FR, RL, RR (Policy Order) or as defined in `train_estimator.py`.
    *   *Note: Raw joint positions are used.*
2.  **Joint Velocities ($\\\dot{q}$)**: 12 dims
    *   *Note: Raw joint velocities are used.*
3.  **Linear Acceleration ($a_{lin}$)**: 3 dims
    *   **CRITICAL DEFINITION**: This represents the **Kinematic Acceleration** in the Body Frame, including the Coriolis/Centrifugal terms.
    *   Formula: $a_{lin} = \\dot{v}_{body} + \\omega_{body} \\times v_{body}$
    *   Deployment Source: `IMU_Accelerometer + Gravity_Vector_Body_Frame`
4.  **Angular Velocity ($\\\omega_{body}$)**: 3 dims
    *   Source: IMU Gyroscope.
5.  **Projected Gravity ($g_{proj}$)**: 3 dims
    *   Gravity vector rotated into the Body Frame (Inverse of body orientation quaternion).

## 4. Training Pipeline

### 4.1 Script Location
`legged_gym/legged_gym/scripts/train_estimator.py`

### 4.2 Data Collection Strategy (Online Rollout)
The training script collects data by running a pre-trained RL policy in the Isaac Gym simulator.

*   **Environment**: `go2_torque` (512 parallel environments)
*   **Policy**: A pre-trained policy checkpoint (e.g., `model_3000.pt`).
*   **Command Randomization**: To ensure dataset diversity (especially for backward motion), commands are randomized every 100 steps:
    *   $v_x \\in [-1.5, 1.5]$ m/s
    *   $v_y \\in [-1.0, 1.0]$ m/s
    *   $\\omega_{yaw} \\in [-1.5, 1.5]$ rad/s

### 4.3 Physics Consistency Fix (Sim vs. Real)
A crucial discrepancy was identified and fixed between the training ground truth calculation and the deployment IMU physics.

*   **Deployment (Real/Sim IMU)**: The IMU measures specific force. When gravity is removed, the remaining term matches **Kinematic Acceleration**: $a_{meas} = \\dot{v} + \\omega \\times v$.
*   **Original Training**: Used simple numerical differentiation: $a_{train} = (v_t - v_{t-1}) / dt \\approx \\dot{v}$.
*   **Fix**: We added the Coriolis term to the training data generation to match the IMU physics:
    ```python
    # In train_estimator.py
    coriolis_acc = torch.cross(current_base_ang_vel, current_base_lin_vel, dim=1)
    lin_acc = (current_base_lin_vel - last_base_lin_vels) / env.dt + coriolis_acc
    ```

### 4.4 Training Parameters
*   **Steps**: 5,000 collection steps (Total samples: ~2.5M)
*   **Epochs**: 20
*   **Batch Size**: 1024
*   **Learning Rate**: 1e-3

## 5. Deployment Architecture

### 5.1 Script Location
`Deploy-an-RL-policy-on-the-Unitree-Go2-robot/src/deploy_rl_policy/scripts/rl_policy.py`

### 5.2 Workflow
1.  **Initialization**: Loads the LSTM model from `resources/policies/velocity_estimator.pt`.
2.  **FSM (Finite State Machine)**:
    *   **CALIBRATE**: Robot lies on the ground. Collects 1000 samples to estimate bias (currently bias subtraction is disabled/zeroed to prevent drift from tilt).
    *   **RUNNING**: Main control loop.
3.  **Feature Construction**:
    ```python
    # Calculate Linear Acceleration from IMU
    lin_acc_raw = imu_accelerometer # (Specific Force)
    pure_lin_acc = lin_acc_raw + (gravity_orientation * 9.81) # Reconstruct Kinematic Acc
    
    features = [q, dq, pure_lin_acc, ang_vel, gravity_orientation]
    ```
4.  **Inference**:
    *   Push `features` to a history buffer (Deque, length 11).
    *   Pass the sequence tensor `(1, 11, 33)` to the LSTM model.
    *   Output `v_est` is used as the observation for the RL policy.

## 6. How to Reproduce

1.  **Train**:
    ```bash
    cd ~/SATA
    python legged_gym/legged_gym/scripts/train_estimator.py --task go2_torque --headless
    ```
2.  **Deploy**:
    Copy the generated model to the deployment folder:
    ```bash
    cp legged_gym/logs/estimator/velocity_estimator_YYYY-MM-DD_HH-MM-SS.pt Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/policies/velocity_estimator.pt
    ```
3.  **Run Sim2Sim**:
    ```bash
    # Terminal 1
    ros2 run deploy_rl_policy mujoco_simulator.py
    # Terminal 2
    ros2 run deploy_rl_policy rl_policy.py --use_velocity_estimator
    # Terminal 3
    ros2 run deploy_rl_policy keyboard_teleop.py
    ```
