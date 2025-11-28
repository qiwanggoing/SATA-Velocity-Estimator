import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GRUVelEstimator(nn.Module):
    """
    State estimator network to estimate the robot's linear velocity.
    
    This network combines a GRU to process temporal information and an MLP
    to output the final velocity, as described in the provided text.
    """
    def __init__(self,
                 input_dim=42,
                 temporal_steps=6, # Example number of historical steps
                 gru_hidden_dim=128,
                 mlp_hidden_dims=[64, 16],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0):
        """
        Initializes the network components.

        Args:
            input_dim (int): Dimension of the input vector for a single timestep (e.g., O_p + prev_action).
            temporal_steps (int): The number of historical timesteps in the input sequence.
            gru_hidden_dim (int): The number of features in the GRU's hidden state.
            mlp_hidden_dims (list): A list of integers for the sizes of the MLP's hidden layers.
            activation (str): The name of the activation function to use in the MLP.
            learning_rate (float): The learning rate for the Adam optimizer.
            max_grad_norm (float): The maximum norm for gradient clipping.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.temporal_steps = temporal_steps
        self.gru_hidden_dim = gru_hidden_dim
        self.max_grad_norm = max_grad_norm
        self.proprio_obs_dim = 45
        self.short_history_length = 6
        self.long_history_length = 100
        
        activation_fn = get_activation(activation)

        # 1. GRU Layer
        # Processes the input sequence. batch_first=True expects
        # input tensors of shape (batch_size, sequence_length, input_dim).
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.gru.flatten_parameters()

        # 2. MLP Head
        # Takes the final hidden state of the GRU as input.
        mlp_layers = []
        current_dim = self.gru_hidden_dim
        for h_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, h_dim))
            mlp_layers.append(activation_fn)
            current_dim = h_dim
        
        # The final layer outputs the 3D linear velocity.
        mlp_layers.append(nn.Linear(current_dim, 3))
        
        self.mlp = nn.Sequential(*mlp_layers)

        # 3. Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs):
        """
        Performs the forward pass of the network.

        Args:
            obs_history (torch.Tensor): A tensor containing the history of observations.
                                        Expected shape: (batch_size, temporal_steps * input_dim)
                                        or (batch_size, temporal_steps, input_dim).

        Returns:
            torch.Tensor: The predicted 3D linear velocity, shape (batch_size, 3).
        """
        # Ensure obs_history is in the sequential format (batch, seq_len, features) for the GRU.
        # if len(obs_history.shape) == 2:
        #     obs_history = obs_history.view(-1, self.temporal_steps, self.input_dim)

        # flat_short_history = obs_history[:, 3+45+51: 3+45+51+self.temporal_steps*42].contiguous()

        B = obs.shape[0]; P = self.proprio_obs_dim
        terrain_one_hot_dim = 1

        expected_obs_dim = 3 + P + terrain_one_hot_dim + \
                           self.short_history_length * P + \
                           self.long_history_length  * P
        assert obs.shape[1] == expected_obs_dim, \
            f"Expected obs.shape[1]={expected_obs_dim}, got {obs.shape[1]}"

        # ---- slices ----
        command_obs = obs[:, :3]                       # not fed to actor directly anymore
        current_obs = obs[:, 3 : 3+P]                # (B, P+3)

        flat_short = obs[:, 3+P+terrain_one_hot_dim : 3+P+terrain_one_hot_dim + self.short_history_length*P]
        short_seq  = flat_short.view(B, self.short_history_length, P)  # includes current step

        flat_long  = obs[:, 3+P+terrain_one_hot_dim + self.short_history_length*P :]
        long_seq   = flat_long.view(B, self.long_history_length,  P)   # continuous past window

        # ---- short latent: project [last-3 + current] to L ----
        # recent_4   = short_seq[:, -4:, :].contiguous().view(B, -1)  # (B, 4P)
        recent_6   = short_seq[:, -6:, 0:42].contiguous()  # (B, 6P)

        # observations_short_history = flat_short_history.reshape(
        #     flat_short_history.shape[0], # batch_size (num_envs)
        #     self.temporal_steps,
        #     42
        # )

        # imu_joint_dim = 30
        # imu_joint_history = observations_short_history[:, :, :imu_joint_dim]

        _, h_n = self.gru(recent_6)

        # Squeeze h_n to remove the num_layers dimension (from 1, batch, hidden) -> (batch, hidden).
        gru_output_for_mlp = h_n.squeeze(0)

        # Pass the GRU's final state through the MLP to get the velocity prediction.
        pred_vel = self.mlp(gru_output_for_mlp)
        
        return pred_vel
    
    def inference(self, obs_history_inference):
        """
        Performs the forward pass of the network.

        Args:
            obs_history (torch.Tensor): A tensor containing the history of observations.
                                        Expected shape: (batch_size, temporal_steps * input_dim)
                                        or (batch_size, temporal_steps, input_dim).

        Returns:
            torch.Tensor: The predicted 3D linear velocity, shape (batch_size, 3).
        """

        _, h_n = self.gru(obs_history_inference.detach())

        # Squeeze h_n to remove the num_layers dimension (from 1, batch, hidden) -> (batch, hidden).
        gru_output_for_mlp = h_n.squeeze(0)

        # Pass the GRU's final state through the MLP to get the velocity prediction.
        pred_vel = self.mlp(gru_output_for_mlp)
        
        return pred_vel

    def update(self, obs_history, critic_obs, lr=None):
        """
        Performs one supervised learning step to update the network weights.

        Args:
            obs_history (torch.Tensor): The input observation history.
            target_vel (torch.Tensor): The ground truth linear velocity for supervised learning.
            lr (float, optional): A new learning rate to set for the optimizer. Defaults to None.

        Returns:
            float: The calculated mean squared error loss for this step.
        """
        if lr is not None and lr != self.learning_rate:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        # Get the predicted velocity from the network.
        pred_vel = self.forward(obs_history)

        target_vel = critic_obs[:, 45 : 45+3].detach()

        # Calculate the Mean Squared Error loss.
        estimation_loss = F.mse_loss(pred_vel, target_vel)
        
        # Perform backpropagation.
        self.optimizer.zero_grad()
        estimation_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item()
    

# --- Helper Function (Unchanged) ---
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu": # Simplified from original
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

