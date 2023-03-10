import torch
from torch import nn

from umt.models.modules.utils import get_clones


class Encoder(nn.Module):
    def __init__(self, layer, depth=1):
        super().__init__()
        self.layers = get_clones(layer, depth)

    def forward(self, x, pos=None, mask=None):
        for layer in self.layers:
            x = layer(x, pos, mask)
        return x
