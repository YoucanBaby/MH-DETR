import torch
import torch.nn.functional as F
from torch import nn

from mh_detr.models.modules.utils import inverse_sigmoid


class Predictor(nn.Module):
    def __init__(self, qkv_dim=256, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.saliency_ffn = nn.Linear(qkv_dim, 1)
        self.span_ffn = nn.Linear(qkv_dim, 2)
        self.score_ffn = nn.Linear(qkv_dim, 2)
        
    def forward(self, qry, mr_qry, mr_ref=None):
        pred_saliency = self.saliency_ffn(qry).squeeze(-1)  #[B, T]
        pred_span = self.span_ffn(mr_qry).sigmoid()         #[B, M, 2]
        pred_span = pred_span + inverse_sigmoid(mr_ref) if mr_ref is not None else pred_span    #[B, M, 2]
        pred_score = self.score_ffn(mr_qry)                 #[B, M, 2]
        
        outputs = {}
        outputs.update(dict(
            saliency = pred_saliency,
            span = pred_span,
            score = pred_score,
        ))
        
        return outputs
