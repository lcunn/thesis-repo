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
from torch.autograd import Variable

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, bidirectional, dropout):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        # Concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hn = hn[-1]
        return hn  # Shape: [batch_size, hidden_size * num_directions]
