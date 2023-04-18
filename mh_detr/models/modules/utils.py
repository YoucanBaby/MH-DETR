import copy

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath
from torch import nn


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def get_activation_fn(activation):
    if activation == "":
        return nn.Identity()
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(f"activation_fn should be relu/gelu, not {activation}.")


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos[:tensor.shape[0]]


def get_key_padding_mask(mask, mask_type, T):
    """
    Args:
        mask: Tensor.size([B, T+N])
    """
    if mask is None:
        return None
    if mask_type == 'vid':
        return mask[:, :T]
    elif mask_type == 'txt':
        return mask[:, T:]
    else:
        raise RuntimeError(f"mask type should be 'vid' or 'txt', not '{mask_type}'.")


def get_clones(module, depth):
    return nn.ModuleList([copy.deepcopy(module) for i in range(depth)])


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', dropout=0.1, num_layers=2):
        super().__init__()
        module = []
        func = get_activation_fn(activation)
        if num_layers == 1:
            module.extend([nn.Linear(input_dim, output_dim), nn.Dropout(dropout)])
        else:
            module.extend([nn.Linear(input_dim, hidden_dim), func, nn.Dropout(dropout)])
            for _ in range(num_layers - 2):
                module.extend([nn.Linear(hidden_dim, hidden_dim), func, nn.Dropout(dropout)])
            module.extend([nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout)])             #最后一层不加激活函数
        self.ffn = nn.Sequential(*module)

    def forward(self, x):
        return self.ffn(x)
    

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(B, N, d)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class SelfAttentionLayer(nn.Module):
    
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu", drop_path=0.):
        super().__init__()
        
        self.sa = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(qkv_dim)

        self.ffn = nn.Sequential(
            nn.Linear(qkv_dim, qkv_dim*4),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(qkv_dim*4, qkv_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(qkv_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
         
    def forward(self, x, pos=None, mask=None):
        """Multi-head Self-Attention Layer
        Args:
            x:      (N, B, d)
            pos:    (N, B, d)
            mask:   (N, B), True is invalid and False is valid.
        Returns:
            x:      (N, B, d)
        """
        
        temp = self.dropout_sa(
            self.sa(
                with_pos_embed(x, pos), 
                with_pos_embed(x, pos), 
                value=x,
                key_padding_mask=mask
            )[0]
        )
        x = x + self.drop_path(temp)
        x = self.norm_sa(x)
        
        temp = self.ffn(x)
        x = x + self.drop_path(temp)
        x = self.norm_ffn(x)
        
        return x
   
   
class CrossAttentionLayer(nn.Module):
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.ca = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_ca = nn.Dropout(dropout)
        self.norm_ca = nn.LayerNorm(qkv_dim)

        self.ffn = nn.Sequential(
            nn.Linear(qkv_dim, qkv_dim*4),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(qkv_dim*4, qkv_dim),
            nn.Dropout(dropout),
        ) 
        self.norm_ffn = nn.LayerNorm(qkv_dim)
         
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):
        temp = self.dropout_ca(
            self.ca(
                with_pos_embed(x, x_pos),
                with_pos_embed(mem, mem_pos),
                value=mem,
                key_padding_mask=mem_mask
            )[0]
        )
        x = x + temp
        x = self.norm_ca(x)
        
        temp = self.ffn(x)
        x = x + temp
        x = self.norm_ffn(x)
        
        return x


class SelfCrossAttentionLayer(nn.Module):
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu", drop_path=0.):
        super().__init__()
        
        self.sa = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(qkv_dim)
        
        self.ca_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
         
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):    
        temp = self.dropout_sa(
            self.sa(
                with_pos_embed(x, x_pos), 
                with_pos_embed(x, x_pos), 
                value=x
            )[0]
        )
        x = x + self.drop_path(temp)
        x = self.norm_sa(x)
        
        x = self.ca_layer(x, mem, x_pos, mem_pos, mem_mask)
        
        return x


class SelfCrossAttentionWithPoolLayer(nn.Module):
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu", drop_path=0.):
        super().__init__()
        
        self.sa = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(qkv_dim)
        
        self.ca_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
         
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):    
        x_before_sa = x
        
        temp = self.dropout_sa(
            self.sa(
                with_pos_embed(x, x_pos), 
                with_pos_embed(x, x_pos), 
                value=x
            )[0]
        )
        x = x + self.drop_path(temp)
        x = self.norm_sa(x)
        
        x = torch.cat((x, x_before_sa),dim=2)
        x = nn.AvgPool1d(kernel_size=2, stride=2)(x)
        
        x = self.ca_layer(x, mem, x_pos, mem_pos, mem_mask)
        
        return x


class SelfCrossAttentionLayerScale(nn.Module):
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu", drop_path=0., 
                                                use_layer_scale=False, layer_scale_init_value=1e-5):
        super().__init__()
        
        # self-attention
        self.sa = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(qkv_dim)
        
        # cross-attention
        self.ca = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_ca = nn.Dropout(dropout)
        self.norm_ca = nn.LayerNorm(qkv_dim)

        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(qkv_dim, qkv_dim*4),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(qkv_dim*4, qkv_dim),
            nn.Dropout(dropout),
        ) 
        self.norm_ffn = nn.LayerNorm(qkv_dim)
        
        # drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        
        # layer_scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_sa = nn.Parameter(
                    layer_scale_init_value * torch.ones((1, 1, qkv_dim)))
            self.layer_scale_ca = nn.Parameter(
                    layer_scale_init_value * torch.ones((1, 1, qkv_dim)))
            self.layer_scale_ffn = nn.Parameter(
                    layer_scale_init_value * torch.ones((1, 1, qkv_dim)))
         
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):    
        # self-attention
        x_before_sa = x    
        
        temp = self.dropout_sa(
            self.sa(
                with_pos_embed(x, x_pos), 
                with_pos_embed(x, x_pos), 
                value=x
            )[0]
        )
        temp = self.layer_scale_sa * temp if self.use_layer_scale else temp
        x = x + self.drop_path(temp)
        x = self.norm_sa(x)
        
        x = torch.cat((x, x_before_sa),dim=2)
        x = nn.AvgPool1d(kernel_size=2, stride=2)(x)
        
        # cross-attention
        temp = self.dropout_ca(
            self.ca(
                with_pos_embed(x, x_pos),
                with_pos_embed(mem, mem_pos),
                value=mem,
                key_padding_mask=mem_mask
            )[0]
        )
        temp = self.layer_scale_ca * temp if self.use_layer_scale else temp
        x = x + self.drop_path(temp)
        x = self.norm_ca(x)
        
        # ffn
        temp = self.ffn(x)
        temp = self.layer_scale_ffn * temp if self.use_layer_scale else temp
        x = x + self.drop_path(temp)
        x = self.norm_ffn(x)
        
        return x


class SelfPoolingLayer(nn.Module):
    
    def __init__(self, qkv_dim=256, pool_size=3, dropout=0.1, activation="relu", drop_path=0.):
        super().__init__()
        
        self.pool = Pooling(pool_size)
        self.norm = nn.LayerNorm(qkv_dim)

        self.ffn = nn.Sequential(
            nn.Linear(qkv_dim, qkv_dim*4),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(qkv_dim*4, qkv_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(qkv_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
         
    def forward(self, x, pos=None, mask=None):
        x = rearrange(x, "T B d -> B d T")
        temp = self.pool(x)
        x = x + self.drop_path(temp)
        x = rearrange(x, "B d T -> T B d")
        x = self.norm(x)
        
        temp = self.ffn(x)
        x = x + self.drop_path(temp)
        x = self.norm_ffn(x)
        
        return x


class SelfPoolingCrossAttentionLayer(nn.Module):
    def __init__(self, qkv_dim=256, num_heads=8, pool_size=3, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.pool = Pooling(pool_size)
        self.norm = nn.LayerNorm(qkv_dim)
        
        self.ca_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
         
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):    
        temp = self.pool(x)
        x = x + temp
        x = self.norm(x)
        
        x = self.ca_layer(x, mem, x_pos, mem_pos, mem_mask)
        
        return x


class WeightedBCE(nn.Module):
    def __init__(self, weight, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target, mask):
        """ Weighted Binary Cross Entropy
        Args:
            output: (B, T)
            target: (B, T)
            mask:   (B, T)
        Returns:
            loss: torch.scalar_tensor
        """
        
        p = torch.clamp(output, min=1e-7, max=1-1e-7)
        
        loss = -(self.weight * target * torch.log(p) + (1 - target) * torch.log(1 - p))
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss
