import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat, reduce, einsum

from umt.models.modules.utils import WeightedBCE
from umt.models.modules.misc import accuracy
from umt.models.modules.matcher import build_matcher
from utils.span_utils import generalized_temporal_iou, span_cxw_to_xx


class UmtCriterion(nn.Module):
    """ This class computes the loss for UMT.
    """
    
    def __init__(self, matcher, weight_dict=None, coef_eos=0.1, max_v_l=75):
        super().__init__()

        self.matcher = matcher
        self.weight_dict = weight_dict
        self.max_v_l = max_v_l
        self.weighted_bce = WeightedBCE(weight=5, reduction="mean")
        
        self.foreground_label = 0
        self.background_label = 1
        self.coef_eos = coef_eos
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.coef_eos  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
    
    
    def get_saliency_loss(self, outputs, targets, indices=None):
        saliency_loss = {}
        saliency_loss.update(self._get_saliency_bce_loss(outputs, targets))
        saliency_loss.update(self._get_saliency_hinge_loss(outputs, targets))
        return saliency_loss
    
    def _get_saliency_bce_loss(self, outputs, targets, indices=None):
        if "saliency_labels" not in targets == 0:
            return {"saliency_bce": 0}
        if "saliency_bce" not in self.weight_dict or self.weight_dict["saliency_bce"] == 0:
            return {"saliency_bce": 0}
        
        saliency_bce = self.weighted_bce(outputs["saliency"], 
                                            targets["saliency_labels"], targets["mask"])
        return {"saliency_bce": saliency_bce}
    
    def _get_saliency_hinge_loss(self, outputs, targets, indices=None):
        if "saliency_pos_labels" not in targets == 0:
            return {"saliency_hinge": 0}
        if "saliency_hinge" not in self.weight_dict or self.weight_dict["saliency_hinge"] == 0:
            return {"saliency_hinge": 0}

        saliency_scores = outputs["saliency"]  #(N, L)
        pos_indices = targets["saliency_pos_labels"]  #(N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  #(N, #pairs)
        num_pairs = pos_indices.shape[1]              #2
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)   #[0, 1... ,B-1]
        
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        
        saliency_hinge_loss = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
            
        return {"saliency_hinge": saliency_hinge_loss}
        
        
    def get_span_loss(self, outputs, targets, indices=None):
        span_loss = {}
        span_loss.update(self._get_span_align_loss(outputs, targets, indices))
        span_loss.update(self._get_span_score_loss(outputs, targets, indices))
        span_loss.update(self._get_span_l1_gious_loss(outputs, targets, indices))
        return span_loss
    
    def _get_span_align_loss(self, outputs, targets, indices=None):
        if "span_align" not in self.weight_dict or self.weight_dict["span_align"] == 0:
            return {"span_align": 0}
        
        B = len(targets["span_labels"])
        mask = targets["mask"]
        
        out_spans = torch.stack([outputs["center"], outputs["window"]], dim=2)
        tgt_spans = targets["spans"]
        
        # TODO 计算out和tgt的IoU
        # generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans)
        
        print(out_spans.shape)
        
        for batch_i in range(B):
            tgt_spans_i = tgt_spans[batch_i]
            print(tgt_spans_i)
        
        return {"span_align": 0}
    
    def _get_span_score_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        src_logits = outputs['score']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'span_score': loss_ce.mean()}

        return losses
        
    def _get_span_l1_gious_loss(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
        The target spans are expected in format (center_x, w), normalized by the image size.
        """
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['span'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)

        span_l1_loss = F.l1_loss(src_spans, tgt_spans, reduction='none')
        span_giou_loss = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))

        losses = {}
        losses['span_l1'] = span_l1_loss.mean()
        losses['span_giou'] = span_giou_loss.mean()
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def forward(self, outputs, targets):
        
        indices = self.matcher(outputs, targets)
        
        losses = {}
        
        saliency_loss = self.get_saliency_loss(outputs, targets, indices)
        span_loss = self.get_span_loss(outputs, targets, indices)
        
        losses.update(saliency_loss)
        losses.update(span_loss)
        
        return losses


def build_criterion(opt):
    """Create loss of UMT"""
    matcher = build_matcher(opt)
    weight_dict = {
                    "saliency_bce": opt.saliency_bce,
                    "saliency_hinge": opt.saliency_hinge,
                    "span_align": opt.span_align,
                    "span_score": opt.span_score,
                    "span_l1": opt.span_l1,
                    "span_giou": opt.span_giou,
                }
    
    criterion = UmtCriterion(
                    matcher, 
                    weight_dict,
                    opt.coef_eos,
                    opt.max_v_l,
    )
    
    device = torch.device(opt.device)
    criterion = criterion.to(device)
    return criterion
