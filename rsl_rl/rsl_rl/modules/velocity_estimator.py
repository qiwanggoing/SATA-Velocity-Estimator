import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GRUVelEstimator(nn.Module):
    """
    State estimator network to estimate the robot's linear velocity.
    """
    def __init__(self,
                 input_dim=45,
                 temporal_steps=6,
                 gru_hidden_dim=128,
                 mlp_hidden_dims=[64, 16],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.temporal_steps = temporal_steps
        self.gru_hidden_dim = gru_hidden_dim
        self.max_grad_norm = max_grad_norm
        
        activation_fn = get_activation(activation)

        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.gru.flatten_parameters()

        # MLP Head
        mlp_layers = []
        current_dim = self.gru_hidden_dim
        for h_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, h_dim))
            mlp_layers.append(activation_fn)
            current_dim = h_dim
        
        # Output 3D linear velocity
        mlp_layers.append(nn.Linear(current_dim, 3))
        
        self.mlp = nn.Sequential(*mlp_layers)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs_history):
        """
        Args:
            obs_history: (batch_size, temporal_steps, input_dim)
        Returns:
            pred_vel: (batch_size, 3)
        """
        # obs_history should be (B, L, D)
        # Pass through GRU
        _, h_n = self.gru(obs_history) # h_n: (num_layers=1, batch, hidden)
        
        gru_out = h_n.squeeze(0) # (batch, hidden)
        
        pred_vel = self.mlp(gru_out)
        return pred_vel

    def update(self, obs_history, target_vel):
        """
        Supervised update step.
        Args:
            obs_history: (batch_size, temporal_steps, input_dim)
            target_vel: (batch_size, 3)
        """
        self.train()
        pred_vel = self.forward(obs_history)
        
        loss = F.mse_loss(pred_vel, target_vel)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
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
