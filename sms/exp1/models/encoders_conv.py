"""
implementing:

a) a convolutional network
- with grid-based data
- with piano-roll based data

b) a bidirectional LSTMs
- with the duration, pitch data

c) a transformer
- with all three types
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict

## convolutional encoders --------------------------------------------------------------------------------------------

# network seen in the siamese CNN for plagiarism paper
ConvPianoRollConfig = {
    "input_size": (128, 64),
    "in_channels": 1,
    "conv_layers": [
        {"out_channels": 16, "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "batch_norm": True},  # Conv2d-1
        {"out_channels": 64, "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "batch_norm": True},  # Conv2d-2
        {"out_channels": 128, "kernel_size": (3, 3), "stride": (2, 2), "padding": (1, 1), "batch_norm": True}, # Conv2d-3
        {"out_channels": 128, "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1), "batch_norm": True}, # Conv2d-4
        {"out_channels": 256, "kernel_size": (3, 3), "stride": (1, 1), "padding": (1, 1), "batch_norm": True}, # Conv2d-5
    ],
    "linear_layers": [
        {"out_features": 4096, "batch_norm": True},  # Linear-1
        {"out_features": 512, "batch_norm": True},   # Linear-2
        {"out_features": 1, "batch_norm": False},    # Linear-3
    ]
}

ConvQuantizedTimeConfig = {
    "input_size": 32,
    "in_channels": 128,
    "conv_layers": [
        {"out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": True},  # Conv1d-1
        {"out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": True},  # Conv1d-2
        {"out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": True},  # Conv1d-3
        {"out_channels": 1024, "kernel_size": 3, "stride": 1, "padding": 1, "batch_norm": True}, # Conv1d-4
    ],
    "linear_layers": [
        {"out_features": 2048, "batch_norm": True},  # Linear-1
        {"out_features": 512, "batch_norm": True},   # Linear-2
        {"out_features": 1, "batch_norm": False},    # Linear-3
    ]
}

def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

class QuantizedConvEncoder(nn.Module):
    def __init__(self, input_size=32, output_size=64):
        super(QuantizedConvEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Define the 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Define fully connected layer to output the final embedding
        self.fc = nn.Linear(64 * (input_size // 8), output_size)
        
        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Assuming input x has shape [batch_size, 1, 32]
        
        # Apply conv layers with ReLU and MaxPooling
        x = self.relu(self.conv1(x))  # [batch_size, 16, 32]
        x = self.pool(x)              # [batch_size, 16, 16]
        
        x = self.relu(self.conv2(x))  # [batch_size, 32, 16]
        x = self.pool(x)              # [batch_size, 32, 8]
        
        x = self.relu(self.conv3(x))  # [batch_size, 64, 8]
        x = self.pool(x)              # [batch_size, 64, 4]
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)     # [batch_size, 64 * 4]
        
        # Apply fully connected layer to produce 64-dimensional embedding
        x = self.fc(x)                # [batch_size, 64]
        
        return x

class PianoRollConvEncoder(nn.Module):
    def __init__(self, input_shape=(128, 32), output_size=64):
        super(PianoRollConvEncoder, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        
        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Define a fully connected layer to produce the output embedding
        self.fc = nn.Linear(64 * (input_shape[0] // 8) * (input_shape[1] // 8), output_size)
        
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Assuming input x has shape [batch_size, 1, 128, 32]
        
        # Apply conv layers with ReLU and MaxPooling
        x = self.relu(self.conv1(x))  # [batch_size, 16, 128, 32]
        x = self.pool(x)              # [batch_size, 16, 64, 16]
        
        x = self.relu(self.conv2(x))  # [batch_size, 32, 64, 16]
        x = self.pool(x)              # [batch_size, 32, 32, 8]
        
        x = self.relu(self.conv3(x))  # [batch_size, 64, 32, 8]
        x = self.pool(x)              # [batch_size, 64, 16, 4]
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)     # [batch_size, 64 * 16 * 4]
        
        # Apply fully connected layer to produce 64-dimensional embedding
        x = self.fc(x)                # [batch_size, 64]
        
        return x