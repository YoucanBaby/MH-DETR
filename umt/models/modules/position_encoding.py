import math
import torch
from torch import nn
from einops import rearrange, repeat, reduce, einsum


class PositionEmbeddingSine(nn.Module):
    def __init__(self, dim=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        """
        Args:
            x:      (B, N, d)
            mask:   (B, N)
        Returns:
            x_pos:  (B, N, d)
        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)   #(B, N)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.dim)

        x_pos = x_embed[:, :, None] / dim_t     #(B, N, d)
        x_pos = torch.stack((x_pos[:, :, 0::2].sin(), x_pos[:, :, 1::2].cos()), dim=3).flatten(2)   #(B, N, d)
        return x_pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num, dim=256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(num, dim))
        nn.init.normal_(self.pos_embed)

    def forward(self, x, mask):
        """
        Args:
            x:      (B, N, d)
            mask:   (B, N)

        Returns:
            x_pos:  (B, N, d)
        """
        B, N, d = x.shape
        mask = rearrange(mask, "b n -> b n 1")
        x_pos = self.pos_embed[:N]                  #[N, d]
        x_pos = repeat(x_pos, "n d -> b n d", b=B)  #[B, N, d]
        x_pos = torch.mul(x_pos, mask)              #[B, N, d]
        return x_pos


class PositionEmbeddingNone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return torch.zeros_like(x)


def  build_position_encoding(type_vid_pos, type_txt_pos, max_v_l, max_q_l, qkv_dim):

    if type_vid_pos == 'sine':
        vid_pos_embed = PositionEmbeddingSine(qkv_dim, normalize=True)
    elif type_vid_pos == 'learned':
        vid_pos_embed = PositionEmbeddingLearned(max_v_l, qkv_dim)
    elif type_vid_pos == 'none':
        vid_pos_embed = PositionEmbeddingNone()
    else:
        raise ValueError(f"not supported {type_vid_pos}")
    
    if type_txt_pos == 'sine':
        txt_pos_embed = PositionEmbeddingSine(qkv_dim, normalize=True)
    elif type_txt_pos == 'learned':
        txt_pos_embed = PositionEmbeddingLearned(max_q_l, qkv_dim)
    elif type_txt_pos == 'none':
        txt_pos_embed = PositionEmbeddingNone()
    else:
        raise ValueError(f"not supported {type_txt_pos}")
    
    return vid_pos_embed, txt_pos_embed
