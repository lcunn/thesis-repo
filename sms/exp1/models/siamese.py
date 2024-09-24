import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from sms.exp1.models.encoders_conv import PianoRollConvEncoder, QuantizedConvEncoder
from sms.exp1.models.encoders_rnn import LSTMEncoder
from sms.exp1.models.encoders_trans import TransformerEncoder
from sms.exp1.models.projector import ProjectionHead

class SiameseModel(nn.Module):
    def __init__(self, config):
        super(SiameseModel, self).__init__()
        
        # Initialize Encoder
        encoder_cfg = config['encoder']
        if encoder_cfg['type'] == 'BidirectionalLSTM':
            self.encoder = LSTMEncoder(
                input_dim=encoder_cfg['params']['input_dim'],
                hidden_size=encoder_cfg['params']['hidden_size'],
                num_layers=encoder_cfg['params']['num_layers'],
                bidirectional=encoder_cfg['params']['bidirectional'],
                dropout=encoder_cfg['params']['dropout']
            )
        elif encoder_cfg['type'] == 'Transformer':
            # Initialize TransformerEncoder
            pass
        elif encoder_cfg['type'] == 'CNN':
            # Initialize CNNEncoder
            pass
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_cfg['type']}")

        # Initialize Projection Head
        projection_cfg = config['projection_head']
        if projection_cfg['type'] == 'MLP':
            self.projection_head = ProjectionHead(
                input_dim=encoder_cfg['params']['hidden_size'] * 
                          (2 if encoder_cfg['params']['bidirectional'] else 1),
                hidden_dims=projection_cfg['params']['hidden_dims'],
                output_dim=projection_cfg['params']['output_dim'],
                dropout=projection_cfg['params']['dropout']
            )
        else:
            raise ValueError(f"Unsupported projection head type: {projection_cfg['type']}")

    def forward_once(self, x):
        encoded = self.encoder(x)
        projected = self.projection_head(encoded)
        return projected

    def forward(self, x1, x2):
        embed1 = self.forward_once(x1)
        embed2 = self.forward_once(x2)
        return embed1, embed2
    

