import argparse
import torch.nn as nn

class SiameseModel(nn.Module):
    def __init__(self, encoder, projector):
        super(SiameseModel, self).__init__()
        
        self.encoder = encoder
        self.projection_head = projector

    def forward_once(self, x):
        encoded = self.encoder(x)
        projected = self.projection_head(encoded)
        return projected

    def forward(self, x1, x2):
        embed1 = self.forward_once(x1)
        embed2 = self.forward_once(x2)
        return embed1, embed2
    
    def get_encoder(self) -> nn.Module:
        return self.encoder