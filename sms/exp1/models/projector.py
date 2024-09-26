import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List

class ProjectionHead(nn.Module):
    def __init__(self, layers: List[Dict], d_latent: int, d_projected: int):
        super().__init__()
        self.d_latent = d_latent
        self.d_projected = d_projected
        
        model_layers = []
        in_features = d_latent
        for layer in layers:
            model_layers.append(nn.Linear(in_features, layer["out_features"]))
            in_features = layer["out_features"]
            model_layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, d_projected))

        self.projector = nn.Sequential(*model_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)