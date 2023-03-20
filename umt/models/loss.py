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
    
    def __init__(self, matcher, weight_dict=None, coef_eos=0.1, temperature=0.07, max_v_l=75):
        super().__init__()

        self.matcher = matcher
        self.weight_dict = weight_dict
        self.max_v_l = max_v_l
        self.weighted_bce = WeightedBCE(weight=5, reduction="mean")
        
        self.foreground_label = 0
        self.background_label = 1
        self.coef_eos = coef_eos
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.coef_eos    #lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        self.temperature = temperature
    
    
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
    
    def _get_saliency_hinge_loss(self, outputs, targets, indices=None, saliency_margin=0.2):
        if "saliency_pos_labels" not in targets == 0:
            return {"saliency_hinge": 0}
        if "saliency_hinge" not in self.weight_dict or self.weight_dict["saliency_hinge"] == 0:
            return {"saliency_hinge": 0}

        saliency_scores = outputs["saliency"]           #(B, T)
        pos_indices = targets["saliency_pos_labels"]    #(N, 2)
        neg_indices = targets["saliency_neg_labels"]    #(N, 2)
        num_pairs = pos_indices.shape[1]                #2
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)   #[0, 1... ,B-1]
        
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        
        saliency_hinge_loss = torch.clamp(saliency_margin + neg_scores - pos_scores, min=0).sum() \
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
        
        if "span_score" not in self.weight_dict or self.weight_dict["span_score"] == 0:
            return {"span_score": 0}
        
        # TODO add foreground and background classifier.  use all non-matched as background.
        src_logits = outputs['score']  #(B, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                                        dtype=torch.int64, device=src_logits.device)  #(B, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'span_score': loss_ce.mean()}

        return losses
        
    def _get_span_l1_gious_loss(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
        The target spans are expected in format (center_x, w), normalized by the image size.
        """
        
        if self.weight_dict["span_l1"] == 0 and self.weight_dict["span_giou"] == 0:
            return {"span_l1": 0, "span_giou": 0}
        
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['span'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)

        span_l1_loss = F.l1_loss(src_spans, tgt_spans, reduction='none')
        span_giou_loss = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))

        losses = {}
        losses['span_l1'] = span_l1_loss.mean() if self.weight_dict["span_l1"] > 0 else 0
        losses['span_giou'] = span_giou_loss.mean() if self.weight_dict["span_giou"] > 0 else 0
        return losses


    def get_contrastive_loss(self, outputs, targets, indices=None):
        contrastive_loss = {}
        contrastive_loss.update(self._get_coarse_contrastive_loss(outputs, targets, indices))
        contrastive_loss.update(self._get_vghd_vg_contrastive_loss(outputs, targets, indices))
        return contrastive_loss
    
    def _get_coarse_contrastive_loss(self, outputs, targets, indices=None):
        if "coarse_contrastive" not in self.weight_dict or self.weight_dict["coarse_contrastive"] == 0:
            return {"coarse_contrastive": 0}
        
        log_softmax = nn.LogSoftmax(dim=-1)
        vid, txt = outputs["vid_pool"], outputs["txt_pool"]     #[B, d], [B, d]

        logits = (txt @ vid.T) / self.temperature
        
        vid_similarity = vid @ vid.T
        txt_similarity = txt @ txt.T
        targets = F.softmax(
            (vid_similarity + txt_similarity) / 2 * self.temperature, dim=-1
        )
        
        vid_loss = (-targets.T * log_softmax(logits.T)).sum(1)
        txt_loss = (-targets * log_softmax(logits)).sum(1)
        
        loss =  (vid_loss + txt_loss) / 2.0     #torch.shape(B)
        
        return {"coarse_contrastive": loss.mean()}

    def _get_vghd_vg_contrastive_loss(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched vghd_qry and input vg_qry"""
        
        if "vghd_vg_contrastive" not in self.weight_dict or self.weight_dict["vghd_vg_contrastive"] == 0:
            return {"vghd_vg_contrastive": 0}
        
        normalized_vghd_embed = outputs["vghd_qry"]     #(B, T, d)
        normalized_vg_embed = outputs["vg_qry"]         #(B, M, d)
        
        logits = torch.einsum(
            "bmd,btd->bmt", normalized_vg_embed, normalized_vghd_embed)     #(B, M, T)
        logits = logits.sum(2) / self.temperature                           #(B, M)
        
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)   #(B, )
        num_pos = positive_map.sum(1)       #(B, )
        neg_term = logits.logsumexp(1)      #(B, )
        
        loss = - pos_term / num_pos + neg_term      #(B, )
        
        return {"vghd_vg_contrastive": loss.mean()}
    
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
        contrastive_loss = self.get_contrastive_loss(outputs, targets, indices)
        
        losses.update(saliency_loss)
        losses.update(span_loss)
        losses.update(contrastive_loss)
        
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
                    "coarse_contrastive": opt.coarse_contrastive,
                    "vghd_vg_contrastive": opt.vghd_vg_contrastive,
                }
    
    criterion = UmtCriterion(
                    matcher, 
                    weight_dict,
                    opt.coef_eos,
                    opt.temperature,
                    opt.max_v_l,
    )
    
    device = torch.device(opt.device)
    criterion = criterion.to(device)
    return criterion
