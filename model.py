# model.py
"""
定义 CNNModelPaper 模型架构。
"""

import torch
import torch.nn as nn


class CNNModelPaper(nn.Module):
    def __init__(self, sequence_length, output_size, dropout_rate):
        """
        初始化模型架构。

        Args:
            sequence_length (int): 输入序列的长度 (例如 90)
            output_size (int): 输出的维度 (例如 1)
            dropout_rate (float): Dropout 的比率
        """
        super(CNNModelPaper, self).__init__()

        self.conv2d_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(in_channels=10 * 3, out_channels=10, kernel_size=12, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=12, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 线性层的输入维度依赖于 sequence_length
        self.fc_base = nn.Sequential(
            nn.Linear(10 * sequence_length, 10),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc_output = nn.Linear(10, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input: (B, 90, 3)
        x = x.unsqueeze(1)  # -> (B, 1, 90, 3)

        x_2d = self.conv2d_layer(x)  # -> (B, 10, 90, 3)

        batch_size, out_channels, seq_len, features_out = x_2d.shape
        x_1d_input = x_2d.view(batch_size, out_channels * features_out, seq_len)  # -> (B, 30, 90)

        x_1d_output = self.conv1d_layers(x_1d_input)  # -> (B, 10, 90)

        x_flat = x_1d_output.reshape(batch_size, -1)  # -> (B, 900)

        x_fc = self.fc_base(x_flat)
        raw_output = self.fc_output(x_fc)

        # 将输出映射到 [70, 100] 的 SpO2 范围
        scaled_output = 70 + 30 * self.sigmoid(raw_output)

        return scaled_output