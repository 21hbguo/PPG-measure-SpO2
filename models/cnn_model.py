# models/cnn_model.py
"""
Defines the CNN (2D+1D Convolution) model architecture.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, sequence_length, output_size, dropout_rate):
        """
        Initializes the CNN model architecture.

        Args:
            sequence_length (int): The length of the input sequence (e.g., 90).
            output_size (int): The dimension of the output (e.g., 1).
            dropout_rate (float): The dropout probability.
        """
        super(CNN, self).__init__()

        # 1. 2D Convolution Layer
        self.conv2d_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 2. 1D Convolution Layers
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(in_channels=10 * 3, out_channels=10, kernel_size=12, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=12, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 3. Fully Connected Layers
        # The input dimension depends on the sequence_length
        self.fc_base = nn.Sequential(
            nn.Linear(10 * sequence_length, 10),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc_output = nn.Linear(10, output_size)

        # Sigmoid for scaling
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (Batch, SequenceLength=90, Features=3)
        x = x.unsqueeze(1)  # -> (B, 1, 90, 3)

        # 2D Convolution
        x_2d = self.conv2d_layer(x)  # -> (B, 10, 90, 3)

        # Reshape for 1D Convolution
        # Combine channels and features
        batch_size, out_channels, seq_len, features_out = x_2d.shape
        x_1d_input = x_2d.view(batch_size, out_channels * features_out, seq_len)  # -> (B, 30, 90)

        # 1D Convolution
        x_1d_output = self.conv1d_layers(x_1d_input)  # -> (B, 10, 90)

        # Flatten for FC layers
        x_flat = x_1d_output.reshape(batch_size, -1)  # -> (B, 900)

        # Fully Connected Layers
        x_fc = self.fc_base(x_flat)
        raw_output = self.fc_output(x_fc)

        # Scale output to SpO2 range [70, 100]
        # scaled_output = 70 + (100 - 70) * sigmoid(raw_output)
        scaled_output = 70 + 30 * self.sigmoid(raw_output)

        return scaled_output