import torch
import torch.nn as nn

class VelocityEstimator(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=128, output_dim=3, num_layers=2):
        """
        A velocity estimator network using an LSTM.
        Matches the architecture trained in legged_gym/scripts/train_estimator.py
        """
        super(VelocityEstimator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # MLP head to regress the velocity
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: The estimated linear velocity of shape (batch_size, output_dim).
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Pass through MLP
        velocity_estimation = self.mlp(last_time_step_out)
        
        return velocity_estimation

if __name__ == '__main__':
    # Example usage for testing
    model = VelocityEstimator()
    print("VelocityEstimator initialized successfully.")
    dummy_input = torch.randn(1, 11, 33)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")