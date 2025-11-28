import torch
import torch.nn as nn

class VelocityEstimator(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=128, output_dim=3, num_layers=2):
        """
        A velocity estimator network using an LSTM.

        Args:
            input_dim (int): The dimension of the input at each time step. 
                             (e.g., q, q_dot, at, wt, gt -> 12+12+3+3+3=33)
            hidden_dim (int): The number of features in the hidden state of the LSTM.
            output_dim (int): The dimension of the output (linear velocity -> 3).
            num_layers (int): The number of recurrent layers in the LSTM.
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
                              Example sequence_length is 11 for t-10 to t.

        Returns:
            torch.Tensor: The estimated linear velocity of shape (batch_size, output_dim).
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # We only need the output of the last time step
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        # hn shape: (num_layers, batch_size, hidden_dim)
        # cn shape: (num_layers, batch_size, hidden_dim)
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Pass the last time step output to the MLP head
        velocity_estimation = self.mlp(last_time_step_out)
        
        return velocity_estimation

if __name__ == '__main__':
    # Example usage:
    batch_size = 32
    sequence_length = 11
    input_dimension = 33
    output_dimension = 3

    # Create dummy input data
    dummy_input = torch.randn(batch_size, sequence_length, input_dimension)

    # Instantiate the estimator
    estimator = VelocityEstimator(input_dim=input_dimension, output_dim=output_dimension)

    # Get the estimated velocity
    estimated_velocity = estimator(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Estimated velocity shape: {estimated_velocity.shape}")
    assert estimated_velocity.shape == (batch_size, output_dimension)
    print("VelocityEstimator created and tested successfully!")

