import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import trunc_normal_

from einops import rearrange, repeat, reduce, einsum

from mh_detr.models.modules.position_encoding import build_position_encoding
from mh_detr.models.modules.encoder import Encoder
from mh_detr.models.modules.decoder import Decoder
from mh_detr.models.modules.utils import (SelfAttentionLayer, CrossAttentionLayer, 
                                      SelfCrossAttentionLayer, SelfCrossAttentionWithPoolLayer, SelfCrossAttentionLayerScale, 
                                      SelfPoolingLayer, SelfPoolingCrossAttentionLayer)


class Backbone(nn.Module):
    
    def __init__(self,  max_v_l=75, max_q_l=32,
                        qkv_dim=256, num_heads=8, 
                        num_mr_qry=10,
                        dropout=0.1, activation="relu", drop_path=0.1,
                        pool_size=3):
        super().__init__()
        
        # Video Encoder
        vid_enc_layer = SelfPoolingLayer(qkv_dim, pool_size, dropout, activation, drop_path)
        self.vid_enc = Encoder(vid_enc_layer, depth=1)
        
        # Text Encoder
        txt_enc_layer = SelfPoolingLayer(qkv_dim, pool_size, dropout, activation, drop_path)
        self.txt_enc = Encoder(txt_enc_layer, depth=1)
        
        # MRHD Query Generator
        qry_gen_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.qry_gen = Decoder(qry_gen_layer, depth=1)
        # MRHD Query Decoder
        qry_dec_layer = SelfCrossAttentionWithPoolLayer(qkv_dim, num_heads, dropout, activation)
        self.qry_dec = Decoder(qry_dec_layer, depth=1)
        
        # MR Decoder
        mr_dec_layer = SelfCrossAttentionWithPoolLayer(qkv_dim, num_heads, dropout, activation)
        self.mr_dec = Decoder(mr_dec_layer, depth=4)
        
        # Query
        self.mr_qry = nn.Parameter(torch.zeros(num_mr_qry, qkv_dim))    #(num_vg_qry, 256)

        # Position Embedding
        self.vid_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(T, 256)
        self.txt_pos = nn.Parameter(torch.zeros(max_q_l, qkv_dim))      #(N, 256)     
        self.qry_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(T, 256)
        
        #Initial Backbone
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, vid, txt, vid_mask, txt_mask):
        B = vid.shape[0]
        
        vid_pos = repeat(self.vid_pos, "T d -> T B d", B=B)
        txt_pos = repeat(self.txt_pos, "N d -> N B d", B=B)
        qry_pos = repeat(self.qry_pos, "T d -> T B d", B=B)
    
        vid, txt = rearrange(vid, "B T d -> T B d"), rearrange(txt, "B N d -> N B d")

        mr_qry = repeat(self.mr_qry, "M d -> M B d", B=B)

        # mask of transformer shuold invert
        vid_mask, txt_mask = ~vid_mask.bool(), ~txt_mask.bool()
        
        vid, txt = self.vid_enc(vid, vid_pos, vid_mask), self.txt_enc(txt, txt_pos, txt_mask)
        
        qry = self.qry_gen(vid, txt, vid_pos, txt_pos, txt_mask)[-1]    #(T, B, d)
        qry = self.qry_dec(qry, vid, qry_pos, vid_pos, vid_mask)[-1]    #(T, B, d)
        
        mr_qry = self.mr_dec(mr_qry, qry, None, qry_pos)[-1]            #(M, B, d)
        
        qry, mr_qry = rearrange(qry, "T B d -> B T d"), rearrange(mr_qry, "M B d -> B M d")
        
        return qry, mr_qry


class BackboneV2(nn.Module):
    
    def __init__(self,  max_v_l=75, max_q_l=32,
                        qkv_dim=256, num_heads=8, 
                        num_mr_qry=10,
                        dropout=0.1, activation="relu", drop_path=0.1,
                        pool_size=3):
        super().__init__()
        
        # Video Encoder
        vid_enc_layer = SelfPoolingLayer(qkv_dim, pool_size, dropout, activation, drop_path)
        self.vid_enc = Encoder(vid_enc_layer, depth=1)
        
        # Text Encoder
        txt_enc_layer = SelfPoolingLayer(qkv_dim, pool_size, dropout, activation, drop_path)
        self.txt_enc = Encoder(txt_enc_layer, depth=1)
        
        # MRHD Query Generator
        qry_gen_layer = CrossAttentionLayer(qkv_dim, num_heads, dropout, activation)
        self.qry_gen = Decoder(qry_gen_layer, depth=1)
        # MRHD Query Decoder
        qry_dec_layer = SelfCrossAttentionWithPoolLayer(qkv_dim, num_heads, dropout, activation)
        self.qry_dec = Decoder(qry_dec_layer, depth=1)
        
        # MR Decoder
        mr_dec_layer = SelfCrossAttentionWithPoolLayer(qkv_dim, num_heads, dropout, activation)
        self.mr_dec = Decoder(mr_dec_layer, depth=4)
        
        # Query
        self.mr_qry = nn.Parameter(torch.zeros(num_mr_qry, qkv_dim))    #(num_vg_qry, 256)
        self.mr_ref = None

        # Position Embedding
        self.vid_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(T, 256)
        self.txt_pos = nn.Parameter(torch.zeros(max_q_l, qkv_dim))      #(N, 256)     
        self.qry_pos = nn.Parameter(torch.zeros(max_v_l, qkv_dim))      #(T, 256)
        
        #Initial Backbone
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, vid, txt, vid_mask, txt_mask):
        """ TODO add dab-detr
        """
        B = vid.shape[0]
        
        vid_pos = repeat(self.vid_pos, "T d -> T B d", B=B)
        txt_pos = repeat(self.txt_pos, "N d -> N B d", B=B)
        qry_pos = repeat(self.qry_pos, "T d -> T B d", B=B)
    
        vid, txt = rearrange(vid, "B T d -> T B d"), rearrange(txt, "B N d -> N B d")

        mr_qry = repeat(self.mr_qry, "M d -> M B d", B=B)

        # mask of transformer shuold invert
        vid_mask, txt_mask = ~vid_mask.bool(), ~txt_mask.bool()
        
        vid, txt = self.vid_enc(vid, vid_pos, vid_mask), self.txt_enc(txt, txt_pos, txt_mask)
        
        qry = self.qry_gen(vid, txt, vid_pos, txt_pos, txt_mask)[-1]    #(T, B, d)
        qry = self.qry_dec(qry, vid, qry_pos, vid_pos, vid_mask)[-1]    #(T, B, d)
        
        mr_qry, mr_ref = self.mr_dec(mr_qry, qry, None, qry_pos)[-1]    #(M, B, d), (M, B, 2)
        
        qry, mr_qry = rearrange(qry, "T B d -> B T d"), rearrange(mr_qry, "M B d -> B M d")
        mr_ref = rearrange(qry, "T B 2 -> B T 2")
        
        return qry, mr_qry, mr_ref
