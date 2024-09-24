import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

default_config = {
    "d_latent": 128,
    "d_projector": 128
}

class ProjectionHead(nn.Module):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if not config:
            config = default_config
        d_latent = config["d_latent"]
        d_projector = config["d_projector"]
        
        self.projector = nn.Sequential(
            nn.Linear(d_latent, d_projector),
            nn.ReLU(),
            nn.Linear(d_projector, d_projector),
            nn.ReLU(),
            nn.Linear(d_projector, d_projector)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)