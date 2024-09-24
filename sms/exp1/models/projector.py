import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict

class ProjectionHead(nn.Module):
    def __init__(self, proj_cfg: Dict, d_latent: int, d_projected: int):
        super().__init__()
        self.d_latent = d_latent
        self.d_projected = d_projected
        
        layers = []
        in_features = d_latent
        for layer in proj_cfg["layers"]:
            layers.append(nn.Linear(in_features, layer["out_features"]))
            in_features = layer["out_features"]
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, d_projected))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)