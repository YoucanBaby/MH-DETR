import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import trunc_normal_

from einops import rearrange, repeat, reduce, einsum

from umt.models.modules.position_encoding import build_position_encoding
from umt.models.modules.encoder import Encoder
from umt.models.modules.decoder import Decoder
from umt.models.modules.utils import SelfAttentionLayer, CrossAttentionLayer, SelfCrossAttentionLayer


class UmtBackbone(nn.Module):
    
    def __init__(self,  max_v_l=75, max_q_l=32,
                        qkv_dim=256, num_heads=8, 
                        num_vg_qry=10,
                        dropout=0.1, activation="relu"):
        super().__init__()
        
        # Video Encoder
        vid_enc_layer = SelfAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.vid_enc = Encoder(vid_enc_layer, depth=1)

        # VGHD Query Generator
        qry_gen_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        # qry_gen_layer = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.qry_gen = Decoder(qry_gen_layer, depth=1)
        
        # VGHD Query Decoder
        qry_dec_layer = SelfCrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.qry_dec = Decoder(qry_dec_layer, depth=1)
        
        # VG Decoder
        vg_dec_layer = SelfCrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.vg_dec = Decoder(vg_dec_layer, depth=4)
        
        # Query
        self.vg_qry = nn.Parameter(torch.zeros(num_vg_qry, qkv_dim))    #(10, 256)

        # Position Embedding
        self.vid_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(75, 256)
        self.txt_pos = nn.Parameter(torch.zeros(max_q_l, qkv_dim))      #(32, 256)     
        self.qry_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(75, 256)
        
        #Initial Backbone
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Parameter):
            trunc_normal_(m, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, vid, txt, vid_mask, txt_mask):
        """
        Args:
            - vid:      [B, T, v_feat_dim]
            - txt:      [B, N, t_feat_dim]
            - vid_mask: [B, T], 0 is invalid and 1 is valid.
            - txt_mask: [B, N], same as vid_mask.
        Returns:
            - vid: [B, T, d]
        """   
        B = vid.shape[0]
        
        vid_pos = repeat(self.vid_pos, "T d -> T B d", B=B)
        txt_pos = repeat(self.txt_pos, "N d -> N B d", B=B)
        qry_pos = repeat(self.qry_pos, "T d -> T B d", B=B)
    
        vid, txt = rearrange(vid, "B T d -> T B d"), rearrange(txt, "B N d -> N B d")

        vg_qry = repeat(self.vg_qry, "M d -> M B d", B=B)

        # mask of transformer shuold invert
        vid_mask, txt_mask = ~vid_mask.bool(), ~txt_mask.bool()
        
        vid = self.vid_enc(vid, vid_pos, vid_mask)
        
        qry = self.qry_gen(vid, txt, vid_pos, txt_pos, txt_mask)    #(T, B, d)
        
        qry = self.qry_dec(qry, vid, qry_pos, vid_pos, vid_mask)    #(T, B, d)
        
        # vg_qry = self.vg_dec(vg_qry, qry, None, qry_pos, vid_mask)  #(M, B, d)        #vg_dec add mask
        vg_qry = self.vg_dec(vg_qry, qry, None, qry_pos)            #(M, B, d)
        
        qry, vg_qry = rearrange(qry, "T B d -> B T d"), rearrange(vg_qry, "M B d -> B M d")
        
        return qry, vg_qry


class UmtV2Backbone(nn.Module):
    
    def __init__(self,  max_v_l=75, max_q_l=32,
                        qkv_dim=256, num_heads=8, 
                        num_vg_qry=10,
                        dropout=0.1, activation="relu"):
        super().__init__()
        
        # Video Encoder
        vid_enc_layer = SelfAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.vid_enc = Encoder(vid_enc_layer, depth=1)

        # VGHD Query Generator
        qry_gen_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        # qry_gen_layer = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.qry_gen = Decoder(qry_gen_layer, depth=1)
        
        # VGHD Query Decoder
        qry_dec_layer = SelfCrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.qry_dec = Decoder(qry_dec_layer, depth=1)
        
        # VG Decoder
        vg_dec_layer = SelfCrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.vg_dec = Decoder(vg_dec_layer, depth=6)
        
        # HD Decoder
        hd_dec_layer = SelfAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.hd_dec = Encoder(hd_dec_layer, depth=1)
        
        # Query
        self.vg_qry = nn.Parameter(torch.zeros(num_vg_qry, qkv_dim))    #(10, 256)

        # Position Embedding
        self.vid_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(75, 256)
        self.txt_pos = nn.Parameter(torch.zeros(max_q_l, qkv_dim))      #(32, 256)     
        self.vg_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))       #(75, 256)
        self.hd_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))       #(75, 256)
        
        #Initial Backbone
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Parameter):
            trunc_normal_(m, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, vid, txt, vid_mask, txt_mask):
        """
        Args:
            - vid:      [B, T, v_feat_dim]
            - txt:      [B, N, t_feat_dim]
            - vid_mask: [B, T], 0 is invalid and 1 is valid.
            - txt_mask: [B, N], same as vid_mask.
        Returns:
            - vid: [B, T, d]
        """   
        B = vid.shape[0]
        
        vid_pos, txt_pos = repeat(self.vid_pos, "T d -> T B d", B=B), repeat(self.txt_pos, "N d -> N B d", B=B)
        vg_pos, hd_pos = repeat(self.vg_pos, "T d -> T B d", B=B), repeat(self.hd_pos, "T d -> T B d", B=B)
    
        vid, txt = rearrange(vid, "B T d -> T B d"), rearrange(txt, "B N d -> N B d")

        vg_qry = repeat(self.vg_qry, "M d -> M B d", B=B)

        # mask of transformer shuold invert
        vid_mask, txt_mask = ~vid_mask.bool(), ~txt_mask.bool()
        
        vid = self.vid_enc(vid, vid_pos, vid_mask)
        
        qry = self.qry_gen(vid, txt, vid_pos, txt_pos, txt_mask)        #(T, B, d)
        
        qry = self.qry_dec(qry, vid, vg_pos+hd_pos, vid_pos, vid_mask)  #(T, B, d)
        
        vg_qry = self.vg_dec(vg_qry, qry, None, vg_pos)                 #(M, B, d)
        hd_qry = self.hd_dec(qry, hd_pos)                               #(T, B, d)
        
        hd_qry, vg_qry = rearrange(hd_qry, "T B d -> B T d"), rearrange(vg_qry, "M B d -> B M d")
        
        return hd_qry, vg_qry


class UmtV3Backbone(nn.Module):
    
    def __init__(self,  max_v_l=75, max_q_l=32,
                        qkv_dim=256, num_heads=8, 
                        num_vg_qry=10,
                        dropout=0.1, activation="relu"):
        super().__init__()
        
        # Video Encoder
        vid_enc_layer = SelfAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.vid_enc = Encoder(vid_enc_layer, depth=1)

        # VGHD Query Generator
        qry_gen_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        # qry_gen_layer = nn.MultiheadAttention(qkv_dim, num_heads, dropout)
        self.qry_gen = Decoder(qry_gen_layer, depth=1)
        
        # VGHD Query Decoder
        qry_dec_layer = SelfCrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.qry_dec = Decoder(qry_dec_layer, depth=1)
        
        # VG Decoder
        vg_dec_layer = SelfCrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.vg_dec = Decoder(vg_dec_layer, depth=6)
        
        # HD Decoder
        hd_dec_layer = SelfAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.hd_dec = Encoder(hd_dec_layer, depth=1)
        
        # Query
        self.vg_qry = nn.Parameter(torch.zeros(num_vg_qry, qkv_dim))    #(10, 256)

        # Position Embedding
        self.vid_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(75, 256)
        self.txt_pos = nn.Parameter(torch.zeros(max_q_l, qkv_dim))      #(32, 256)     
        self.vg_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))       #(75, 256)
        self.hd_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))       #(75, 256)
        
        #Initial Backbone
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Parameter):
            trunc_normal_(m, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, vid, txt, vid_mask, txt_mask):
        """
        Args:
            - vid:      [B, T, v_feat_dim]
            - txt:      [B, N, t_feat_dim]
            - vid_mask: [B, T], 0 is invalid and 1 is valid.
            - txt_mask: [B, N], same as vid_mask.
        Returns:
            - vid: [B, T, d]
        """   
        B = vid.shape[0]
        
        vid_pos, txt_pos = repeat(self.vid_pos, "T d -> T B d", B=B), repeat(self.txt_pos, "N d -> N B d", B=B)
        vg_pos, hd_pos = repeat(self.vg_pos, "T d -> T B d", B=B), repeat(self.hd_pos, "T d -> T B d", B=B)
    
        vid, txt = rearrange(vid, "B T d -> T B d"), rearrange(txt, "B N d -> N B d")

        vg_qry = repeat(self.vg_qry, "M d -> M B d", B=B)

        # mask of transformer shuold invert
        vid_mask, txt_mask = ~vid_mask.bool(), ~txt_mask.bool()
        
        vid = self.vid_enc(vid, vid_pos, vid_mask)
        
        qry = self.qry_gen(vid, txt, vid_pos, txt_pos, txt_mask)        #(T, B, d)
        
        qry = self.qry_dec(qry, vid, vg_pos+hd_pos, vid_pos, vid_mask)  #(T, B, d)
        
        vg_qry = self.vg_dec(vg_qry, qry, None, vg_pos)                 #(M, B, d)
        hd_qry = self.hd_dec(qry, hd_pos)                               #(T, B, d)
        
        hd_qry, vg_qry = rearrange(hd_qry, "T B d -> B T d"), rearrange(vg_qry, "M B d -> B M d")
        
        return hd_qry, vg_qry