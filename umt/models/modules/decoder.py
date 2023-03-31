import torch
from torch import nn

from umt.models.modules.utils import get_clones, with_pos_embed


class Decoder(nn.Module):
    def __init__(self, layer, depth=1):
        super().__init__()
        self.layers = get_clones(layer, depth)
        
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):
        res = []
        if isinstance(self.layers[0], nn.MultiheadAttention):
            for layer in self.layers:
                x = layer(
                        with_pos_embed(x, x_pos),
                        with_pos_embed(mem, mem_pos),
                        value=mem,
                        key_padding_mask=mem_mask
                )[0]
        else:
            for layer in self.layers:
                x = layer(x, mem, x_pos, mem_pos, mem_mask)
                res.append(x)
        return res
