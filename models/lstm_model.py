# models/lstm_model.py
"""
Defines a simple LSTM model architecture (example).
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): Input feature dimension (e.g., 3 for R, G, B).
            hidden_size (int): Size of LSTM hidden state.
            output_size (int): Output dimension (e.g., 1 for SpO2).
            num_layers (int): Number of LSTM layers.
            dropout_rate (float): Dropout probability.
        """
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape (B, SeqLen, Features)
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.fc_base = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.fc_output = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (B, 90, 3)

        # LSTM output: (B, 90, hidden_size)
        # We only care about the output, not the hidden state (h_n, c_n)
        lstm_out, _ = self.lstm(x)

        # Use the output of the last time step for prediction
        last_time_step_out = lstm_out[:, -1, :]  # -> (B, hidden_size)

        # Fully Connected Layers
        x_fc = self.fc_base(last_time_step_out)
        raw_output = self.fc_output(x_fc)

        # Scale output to SpO2 range [70, 100]
        scaled_output = 70 + 30 * self.sigmoid(raw_output)

        return scaled_output