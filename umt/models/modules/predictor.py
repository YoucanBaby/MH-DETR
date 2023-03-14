import torch
import torch.nn.functional as F
from torch import nn


class UmtPredictor(nn.Module):
    def __init__(self, qkv_dim=256, dropout=0.1, activation="relu", offset=False):
        super().__init__()
        
        self.offset = offset
        
        self.saliency_ffn = nn.Linear(qkv_dim, 1)
        self.span_ffn = nn.Linear(qkv_dim, 2)
        self.score_ffn = nn.Linear(qkv_dim, 2)
        
    def forward(self, qry, vg_qry):
        # pred_saliency = self.saliency_ffn(qry).squeeze(-1).sigmoid()    #[B, T]
        pred_saliency = self.saliency_ffn(qry).squeeze(-1)  #[B, T]
        pred_span = self.span_ffn(vg_qry).sigmoid()         #[B, M, 2]
        pred_score = self.score_ffn(vg_qry)                 #[B, M, 2]
        # pred_score = self.score_ffn(vg_qry).sigmoid()                   #[B, M, 2]
        
        outputs = {}
        outputs.update(dict(
            saliency = pred_saliency,
            span = pred_span,
            score = pred_score,
        ))
        
        return outputs
