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

# ALL ENCODERS SHOULD HAVE FORWARD METHODS THAT TAKE TENSORS OF SHAPE [batch_size, *input_shape] AND RETURN TENSORS OF SHAPE [batch_size, d_latent].

## convolutional encoders --------------------------------------------------------------------------------------------

def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

class QuantizedConvEncoder(nn.Module):
    def __init__(self, layers: List[Dict], input_shape: int = 32, d_latent: int = 64):
        """
        A convolutional encoder for quantized data.
        Expects a 1D input of shape [batch_size, input_shape].

        Requires formatting:
        


        Each layer requires the following parameters:
        - kernel_size: int
        - stride: int
        - padding: int
        - out_channels: int
        - batch_norm: bool
        - max_pool: bool
        """
        super(QuantizedConvEncoder, self).__init__()
        self.input_shape = input_shape
        self.d_latent = d_latent

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
            if layer_cfg.get('max_pool', False):
                model_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = layer_cfg['out_channels']
        
        self.conv_layers = nn.Sequential(*model_layers)
        
        # calculate the size of the output from the conv layers
        conv_output_size = self._get_conv_output_size(self.input_shape, layers)
        
        self.fc = nn.Linear(conv_output_size, d_latent)

    def _get_conv_output_size(self, input_size, conv_layers):
        size = input_size
        for layer_cfg in conv_layers:
            size = calculate_conv_output_size(size, layer_cfg['kernel_size'], layer_cfg['stride'], layer_cfg['padding'])
            # Uncomment the following line if you're using max pooling
            # size //= 2
        return size * conv_layers[-1]['out_channels']

    def forward(self, batch):
        # assuming input batch has shape [batch_size, input_shape]
        batch = batch.unsqueeze(1)            # [batch_size, 1, input_shape]
        batch = self.conv_layers(batch)
        batch = batch.view(batch.size(0), -1)     # [batch_size, conv_output_size]
        batch = self.fc(batch)                # [batch_size, d_latent]
        return batch

class PianoRollConvEncoder(nn.Module):
    def __init__(self, layers: List[Dict], input_shape: Tuple[int, int] = (128, 32), d_latent: int = 64):
        """
        A convolutional encoder for piano-roll data.

        Requires formatting:
        
        Each layer requires the following parameters:
        - kernel_size: tuple[int, int]
        - stride: tuple[int, int]
        - padding: tuple[int, int]
        - out_channels: int
        - batch_norm: bool
        - max_pool: bool
        """
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
            if layer_cfg.get('max_pool', False):
                model_layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
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

    def forward(self, batch):
        # (assuming input batch has shape [batch_size, 128, 32])
        # add channel dimension
        batch = batch.unsqueeze(1)
        batch = self.conv_layers(batch)
        # flatten tensor
        batch = batch.view(batch.size(0), -1)     # [batch_size, conv_output_size]
        # apply fc layer
        batch = self.fc(batch)                # [batch_size, d_latent]
        return batch
    
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

### sequential data

class TokenAndPositionalEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, emb_dim, max_len):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.token_emb = nn.Conv1d(self.input_dim, self.emb_dim, 1)
        self.pos_emb = self.positional_encoding(self.max_len, self.emb_dim)

    def get_angles(self, pos, i, emb_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(emb_dim))
        return pos * angle_rates

    def positional_encoding(self, position, emb_dim):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(emb_dim)[np.newaxis, :],
            emb_dim,
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return torch.tensor(pos_encoding, dtype=torch.float32)

    def forward(self, x):
        if x.dim() == 2:
            # If input is 2D (batch_size, seq_len), add a dimension
            x = x.unsqueeze(-1)
        
        x = torch.permute(x, (0, 2, 1))
        x = self.token_emb(x)
        x *= torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float32))
        x = torch.permute(x, (0, 2, 1))
        return x + self.pos_emb.to(x.device)[:, : x.shape[1]]

class BertEncoder(nn.Module):
    def __init__(self, config, input_shape=2, d_latent=64, pad_value=-1000):
        """
        BERT encoder for sequential input.

        Args:
            config (dict): Configuration dictionary with the following keys:
                - d_model (int): The dimension of the model's hidden states. Default is 128.
                - n_layers (int): The number of transformer encoder layers. Default is 4.
                - max_seq_len (int): The maximum sequence length. Default is 50.
                - n_heads (int): The number of attention heads in each layer. Default is 8.
                - d_ff (int): The dimension of the feedforward network model. Default is d_model * 4.
                - dropout_rate (float): The dropout rate. Default is 0.1.

            input_shape (int): The dimension of each input token. Default is 2.
            d_latent (int): The dimension of the output latent representation. Default is 64.
            pad_value (float): The value used for padding. Default is -1000.
        """
        super(BertEncoder, self).__init__()
        self.d_input = input_shape
        self.d_latent = d_latent
        self.d_model = config.get("d_model", 128)
        self.n_layers = config.get("n_layers", 4)
        self.pad_value = pad_value  # Added pad_value parameter

        self.emb = TokenAndPositionalEmbeddingLayer(
            input_dim=self.d_input, emb_dim=self.d_model, max_len=config.get("max_seq_len", 50)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get("n_heads", 8),
            dim_feedforward=config.get("d_ff", self.d_model * 4),
            dropout=config.get("dropout_rate", 0.1),
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )
        self.fc = nn.Linear(self.d_model, self.d_latent)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, batch):
        # assume input batch has shape [batch_size, padded_seq_length, point_dim]
        
        # create padding mask where all features are equal to pad_value
        batch_key_padding_mask = torch.all(batch == self.pad_value, dim=-1)
        batch_key_padding_mask = batch_key_padding_mask.to(batch.device)
        
        batch_emb = self.emb(batch)             # (batch_size, padded_seq_length, d_model)

        if not bool(torch.sum(batch_key_padding_mask)):
            batch_key_padding_mask = None # prevents next line breaking
        batch_emb = self.transformer_encoder(
            batch_emb, src_key_padding_mask=batch_key_padding_mask  # Correct parameter name
        )                                       # (batch_size, padded_seq_length, d_model)
        batch_emb = self.fc(batch_emb)          # (batch_size, padded_seq_length, d_latent)
        batch_emb = torch.permute(batch_emb, (0, 2, 1))  # (batch_size, d_latent, padded_seq_length)
        batch_emb = self.pool(batch_emb)            # (batch_size, d_latent, 1)
        batch_emb = torch.squeeze(batch_emb, dim=2)  # (batch_size, d_latent)

        return batch_emb
