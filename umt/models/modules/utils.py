import copy

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import nn


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


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
    
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu"):
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
        x = x + temp
        x = self.norm_sa(x)
        
        temp = self.ffn(x)
        x = x + temp
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
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.sa = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(qkv_dim)
        
        self.ca_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
         
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):    
        temp = self.dropout_sa(
            self.sa(
                with_pos_embed(x, x_pos), 
                with_pos_embed(x, x_pos), 
                value=x
            )[0]
        )
        x = x + temp
        x = self.norm_sa(x)
        
        x = self.ca_layer(x, mem, x_pos, mem_pos, mem_mask)
        
        return x
    
    
class SelfCrossAttentionWithPoolLayer(nn.Module):
    def __init__(self, qkv_dim=256, num_heads=8, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.sa = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(qkv_dim)
        
        self.ca_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
         
    def forward(self, x, mem, x_pos=None, mem_pos=None, mem_mask=None):    
        T = x
        temp = self.dropout_sa(
            self.sa(
                with_pos_embed(x, x_pos), 
                with_pos_embed(x, x_pos), 
                value=x
            )[0]
        )
        
        x = x + temp
        x = self.norm_sa(x)
        
        x = torch.cat((x,T),dim=2)
        x = nn.AvgPool1d(kernel_size=2, stride=2)(x)
        
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


if __name__ == '__main__':
    # 测试masked_fill_(), masked_fill()不更新变量, masked_fill_()更新变量
    bce = WeightedBCE()
    
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    output = torch.tensor([[1, 1, 1], [1, 0.5, 0]])
    target = torch.tensor([[1, 1, 0], [1, 0, 0]])
    
    loss = bce(output, target, mask)
    
    print(loss)
    