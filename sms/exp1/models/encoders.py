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
from typing import Optional, Dict, Tuple, List

def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

## convolutional encoders --------------------------------------------------------------------------------------------

class QuantizedConvEncoder(nn.Module):
    def __init__(self, layers: List[Dict], input_shape: int = 32, d_latent: int = 64):
        super(QuantizedConvEncoder, self).__init__()
        self.input_shape = input_shape
        self.d_latent = d_latent
        
        # Define the CNN layers based on the config
        model_layers = []
        in_channels = 1
        for layer_cfg in layers:
            model_layers.append(nn.Conv1d(in_channels=in_channels, 
                                    out_channels=layer_cfg['out_channels'], 
                                    kernel_size=layer_cfg['kernel_size'], 
                                    stride=layer_cfg['stride'], 
                                    padding=layer_cfg['padding']))
            if layer_cfg.get('batch_norm', False):
                model_layers.append(nn.BatchNorm1d(layer_cfg['out_channels']))
            model_layers.append(nn.ReLU())
            # layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = layer_cfg['out_channels']
        
        self.conv_layers = nn.Sequential(*model_layers)
        
        # Calculate the size of the output from the conv layers
        conv_output_size = self._get_conv_output_size(self.input_shape, layers)
        
        # Define a fully connected layer to produce the output embedding
        self.fc = nn.Linear(conv_output_size, d_latent)

    def _get_conv_output_size(self, input_size, conv_layers):
        size = input_size
        for layer_cfg in conv_layers:
            size = calculate_conv_output_size(size, layer_cfg['kernel_size'], layer_cfg['stride'], layer_cfg['padding'])
            # size //= 2  # MaxPool1d
        return size * conv_layers[-1]['out_channels']

    def forward(self, x):
        # assuming input x has shape [batch_size, 1, 32]
        # apply conv layers
        # print(f"Input x requires_grad: {x.requires_grad}")
        x = self.conv_layers(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)     # [batch_size, conv_output_size]
        # apply fully connected layer to produce 64-dimensional embedding
        x = self.fc(x)                # [batch_size, 64]
        # print(f"Output x requires_grad: {x.requires_grad}")
        return x

class PianoRollConvEncoder(nn.Module):
    def __init__(self, layers: List[Dict], input_shape: Tuple[int, int] = (128, 32), d_latent: int = 64):
        super(PianoRollConvEncoder, self).__init__()
        self.input_shape = input_shape
        self.d_latent = d_latent
        
        # Define the CNN layers based on the config
        model_layers = []
        in_channels = 1
        for layer_cfg in layers:
            model_layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=layer_cfg['out_channels'], 
                                    kernel_size=tuple(layer_cfg['kernel_size']), 
                                    stride=tuple(layer_cfg['stride']), 
                                    padding=tuple(layer_cfg['padding'])))
            if layer_cfg.get('batch_norm', False):
                model_layers.append(nn.BatchNorm2d(layer_cfg['out_channels']))
            model_layers.append(nn.ReLU())
            # model_layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            in_channels = layer_cfg['out_channels']
        
        self.conv_layers = nn.Sequential(*model_layers)
        
        # Calculate the size of the output from the conv layers
        conv_output_size = self._get_conv_output_size(input_shape, layers)
        
        # Define a fully connected layer to produce the output embedding
        self.fc = nn.Linear(conv_output_size, d_latent)

    def _get_conv_output_size(self, input_shape, conv_layers):
        h, w = input_shape
        for layer_cfg in conv_layers:
            h = calculate_conv_output_size(h, layer_cfg['kernel_size'][0], layer_cfg['stride'][0], layer_cfg['padding'][0])
            w = calculate_conv_output_size(w, layer_cfg['kernel_size'][1], layer_cfg['stride'][1], layer_cfg['padding'][1])
            # h //= 2  # MaxPool2d
            # w //= 2  # MaxPool2d
        return h * w * conv_layers[-1]['out_channels']

    def forward(self, x):
        # Assuming input x has shape [batch_size, 1, 128, 32]
        # Apply conv layers
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)     # [batch_size, conv_output_size]
        # Apply fully connected layer to produce 64-dimensional embedding
        x = self.fc(x)                # [batch_size, 64]
        return x
    
## bidirectional LSTMs --------------------------------------------------------------------------------------------

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
    
## transformer encoders --------------------------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        pass
