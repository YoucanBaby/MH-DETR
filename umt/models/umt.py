import torch
import torch.nn.functional as F
from torch import nn

import timm
from timm.models.layers import trunc_normal_

from einops import rearrange, repeat, reduce, einsum

from umt.models.modules.input_ffn import InputFFN
from umt.models.modules.backbone import UmtBackbone
from umt.models.modules.predictor import UmtPredictor


class UMT(nn.Module):
    def __init__(self, backbone, v_feat_dim, t_feat_dim, 
                        input_vid_ffn_dropout=0.5, input_txt_ffn_dropout=0.3, qkv_dim=256, dropout=0.1):
        super().__init__()
       
        self.input_ffn = InputFFN(v_feat_dim, t_feat_dim, qkv_dim, input_vid_ffn_dropout, input_txt_ffn_dropout)
        self.backbone = backbone
        self.predictor = UmtPredictor(qkv_dim, dropout)

    def forward(self, vid, txt, vid_mask, txt_mask):
        """The forward expects four tensors:
            - vid:      [B, T, v_feat_dim]
            - txt:      [B, N, t_feat_dim]
            - vid_mask: [B, T], 0 is invalid and 1 is valid.
            - txt_mask: [B, N], same as vid_mask.
        """
        vid, txt = self.input_ffn(vid, txt)                 #[B, T, d], [B, N, d]
        qry, vg_qry = self.backbone(vid, txt, vid_mask, txt_mask)   #[B, T, d], [B, M, d]
        outputs = self.predictor(qry, vg_qry)
        return outputs


def build_umt(opt):
    """Create model of UMT"""
    umt_backbone = UmtBackbone(
        opt.max_v_l, opt.max_q_l,
        opt.qkv_dim, opt.num_heads, 
        opt.num_vg_qry,
        opt.dropout, opt.activation, 
    )
    model = UMT(
        umt_backbone, 
        opt.v_feat_dim, opt.t_feat_dim, 
        opt.input_vid_ffn_dropout, opt.input_txt_ffn_dropout,
        opt.qkv_dim, 
        opt.dropout,
    )
    return model