import argparse
import torch.nn as nn

# class SiameseModel(nn.Module):
#     def __init__(self, encoder, projector):
#         super(SiameseModel, self).__init__()
        
#         self.encoder = encoder
#         self.projection_head = projector

#     def forward_once(self, x):
#         encoded = self.encoder(x)
#         projected = self.projection_head(encoded)
#         return projected

#     def forward(self, x1, x2):
#         embed1 = self.forward_once(x1)
#         embed2 = self.forward_once(x2)
#         return embed1, embed2
    
#     def get_encoder(self) -> nn.Module:
#         return self.encoder

class SiameseModel(nn.Module):
    def __init__(self, encoder, projector):
        super(SiameseModel, self).__init__()
        
        self.encoder = encoder
        self.projection_head = projector
        self.use_projection = True

    def forward_once(self, x):
        encoded = self.encoder(x)
        if self.use_projection:
            projected = self.projection_head(encoded)
            return projected
        return encoded

    # def forward(self, x1, x2, x3=None):
    #     embed1 = self.forward_once(x1)
    #     embed2 = self.forward_once(x2)
    #     if x3 is not None:
    #         embed3 = self.forward_once(x3)
    #         return embed1, embed2, embed3
    #     return embed1, embed2   
    
    def forward(self, *args):
        embeds = [self.forward_once(x) for x in args]
        return embeds[0] if len(embeds) == 1 else tuple(embeds)  # Return single tensor or tuple
    
    def get_encoder(self) -> nn.Module:
        return self.encoder

    def set_use_projection(self, use_projection: bool):
        self.use_projection = use_projection