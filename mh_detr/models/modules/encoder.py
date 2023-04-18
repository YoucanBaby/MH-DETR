import torch
from torch import nn

from mh_detr.models.modules.utils import get_clones


class Encoder(nn.Module):
    def __init__(self, layer, depth=1):
        super().__init__()
        self.depth = depth
        self.layers = get_clones(layer, depth)

    def forward(self, x, pos=None, mask=None):
        if self.depth == 0:
            return x
        for layer in self.layers:
            x = layer(x, pos, mask)
        return x
