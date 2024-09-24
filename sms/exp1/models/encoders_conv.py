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
from typing import Optional, Dict, Tuple

def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

## convolutional encoders --------------------------------------------------------------------------------------------

class QuantizedConvEncoder(nn.Module):
    def __init__(self, encoder_cfg: Dict, input_length: int = 32, output_size: int = 64):
        super(QuantizedConvEncoder, self).__init__()
        self.input_length = input_length
        self.output_size = output_size
        
        # Define the CNN layers based on the config
        layers = []
        in_channels = 1
        for layer_cfg in encoder_cfg['layers']:
            layers.append(nn.Conv1d(in_channels=in_channels, 
                                    out_channels=layer_cfg['out_channels'], 
                                    kernel_size=layer_cfg['kernel_size'], 
                                    stride=layer_cfg['stride'], 
                                    padding=layer_cfg['padding']))
            if layer_cfg.get('batch_norm', False):
                layers.append(nn.BatchNorm1d(layer_cfg['out_channels']))
            layers.append(nn.ReLU())
            # layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = layer_cfg['out_channels']
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size of the output from the conv layers
        conv_output_size = self._get_conv_output_size(self.input_length, encoder_cfg['layers'])
        
        # Define a fully connected layer to produce the output embedding
        self.fc = nn.Linear(conv_output_size, output_size)

    def _get_conv_output_size(self, input_size, conv_layers):
        size = input_size
        for layer_cfg in conv_layers:
            size = calculate_conv_output_size(size, layer_cfg['kernel_size'], layer_cfg['stride'], layer_cfg['padding'])
            # size //= 2  # MaxPool1d
        return size * conv_layers[-1]['out_channels']

    def forward(self, x):
        # assuming input x has shape [batch_size, 1, 32]
        # apply conv layers
        x = self.conv_layers(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)     # [batch_size, conv_output_size]
        # apply fully connected layer to produce 64-dimensional embedding
        x = self.fc(x)                # [batch_size, 64]
        
        return x

class PianoRollConvEncoder(nn.Module):
    def __init__(self, encoder_cfg: Dict, input_shape: Tuple[int, int] = (128, 32), output_size: int = 64):
        super(PianoRollConvEncoder, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        
        # Define the CNN layers based on the config
        layers = []
        in_channels = 1
        for layer_cfg in encoder_cfg['conv_layers']:
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=layer_cfg['out_channels'], 
                                    kernel_size=tuple(layer_cfg['kernel_size']), 
                                    stride=tuple(layer_cfg['stride']), 
                                    padding=tuple(layer_cfg['padding'])))
            if layer_cfg.get('batch_norm', False):
                layers.append(nn.BatchNorm2d(layer_cfg['out_channels']))
            layers.append(nn.ReLU())
            # layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            in_channels = layer_cfg['out_channels']
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size of the output from the conv layers
        conv_output_size = self._get_conv_output_size(input_shape, encoder_cfg['conv_layers'])
        
        # Define a fully connected layer to produce the output embedding
        self.fc = nn.Linear(conv_output_size, output_size)

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
        x = self.conv_layers(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)     # [batch_size, conv_output_size]
        # Apply fully connected layer to produce 64-dimensional embedding
        x = self.fc(x)                # [batch_size, 64]
        return x