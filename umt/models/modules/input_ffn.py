import torch
import torch.nn.functional as F
from torch import nn


from umt.models.modules.utils import LinearLayer


class InputFFN(nn.Module):
    def __init__(self, v_feat_dim, t_feat_dim, qkv_dim, input_vid_ffn_dropout, input_txt_ffn_dropout):
        super().__init__()
        self.input_video_ffn = nn.Sequential(
            LinearLayer(v_feat_dim, qkv_dim, layer_norm=True, dropout=input_vid_ffn_dropout, relu=True),
            LinearLayer(qkv_dim, qkv_dim, layer_norm=True, dropout=input_vid_ffn_dropout, relu=True)
        )
        self.input_text_ffn = nn.Sequential(
            LinearLayer(t_feat_dim, qkv_dim, layer_norm=True, dropout=input_txt_ffn_dropout, relu=True),
            LinearLayer(qkv_dim, qkv_dim, layer_norm=True, dropout=input_txt_ffn_dropout, relu=True)
        )
    
    def forward(self, vid, txt):
        """
        Args:
            vid: (B, T, v_feat_dim)
            txt: (B, N, t_feat_dim)
        Returns:
            vid: (B, T, qkv_dim)
            txt: (B, N, qkv_dim)
        """
        vid = self.input_video_ffn(vid)
        txt = self.input_text_ffn(txt)
        return vid, txt
    