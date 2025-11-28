# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import torch

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2TorqueCfg(GO2RoughCfg):
    class env(GO2RoughCfg.env):
        num_envs = 1024
        num_observations = 60
        num_actions = 12
        episode_length_s = 10

    class init_state(GO2RoughCfg.init_state):
        pos = [0.0, 0.0, 0.10]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 1.45,  # [rad]
            'RL_thigh_joint': 1.45,  # [rad]
            'FR_thigh_joint': 1.45,  # [rad]
            'RR_thigh_joint': 1.45,  # [rad]

            'FL_calf_joint': -2.5,  # [rad]
            'RL_calf_joint': -2.5,  # [rad]
            'FR_calf_joint': -2.5,  # [rad]
            'RR_calf_joint': -2.5,  # [rad]
        }

    class asset(GO2RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf'
        self_collisions = 0
        terminate_after_contacts_on = ["Head"]
        penalize_contacts_on = ["thigh", "calf"]

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 1  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.8, 0, 0, 0.0]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands(GO2RoughCfg.commands):
        heading_command = False  # if true: compute ang vel command from heading error
        resampling_time = 5

        class ranges:
            lin_vel_x = [-0.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class control(GO2RoughCfg.control):
        control_type = 'TG' # 'T': torque control, 'TG': torque control with growth
        activation_process = True
        hill_model = True
        motor_fatigue = True
        action_scale = 5
        decimation = 1

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.2
            height_measurements = 0.1
            fatigue = 0.5

    class rewards(GO2RoughCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25
        base_height_target = 0.3
        max_contact_force = 100.

        class scales:
            forward = 10.
            head_height = 5.
            moving_y = 5.
            moving_yaw = 5.

            soft_dof_pos_limits = -5.0
            motor_fatigue = -0.05
            dof_acc = -1e-6
            roll = -5.
            lin_vel_z = -5.

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 5.]
        shifted_com_range_x = [-0.2, 0.2]
        shifted_com_range_y = [-0.1, 0.1]
        shifted_com_range_z = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 1.5
        max_push_vel_ang = 1.0
        loss_action_obs = True
        loss_rate = 0.1

    class growth:
        max_torque_scale = 1.0
        start_torque_scale = 0.3
        max_rear_torque_scale = 1.0
        start_rear_torque_scale = 1.0

        max_freq = 200
        start_freq = 100

        k = 0.00003
        x0 = 1000 * 24

    class test:
        use_test = False
        checkpoint = 3000
        vel = torch.tensor([1.0, 0.0, 0.0, 0.], dtype=torch.float32)


class GO2TorqueCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        run_name = ''
        experiment_name = 'SATA'
        max_iterations = 3000