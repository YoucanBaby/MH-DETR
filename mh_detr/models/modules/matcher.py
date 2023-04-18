import torch
from torch import nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from utils.span_utils import generalized_temporal_iou, span_cxw_to_xx


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

        For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, span_score=1, span_l1=1, span_giou=1, span_loss_type="l1", max_v_l=75) -> None:
        super().__init__()
        self.span_score = span_score
        self.span_l1 = span_l1
        self.span_giou = span_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert span_score != 0 or span_l1 != 0 or span_giou != 0  #不能全为0
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        
        B, M, d = outputs["span"].shape
        targets = targets["span_labels"]

        # Also concat the target labels and spans
        out_score = outputs["score"].flatten(0, 1).softmax(-1)          #[B * num_queries, num_classes==2]
        tgt_spans = torch.cat([v["spans"] for v in targets])            #[num_target_spans in batch, 2]
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   #[total #spans in the batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_score[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["span"].flatten(0, 1)     # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            # [batch_size * num_queries, total #spans in the batch]
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            pred_spans = outputs["span"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(B * M, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # giou
            cost_giou = 0

        # Final cost matrix
        # import ipdb; ipdb.set_trace()
        C = self.span_l1 * cost_span + self.span_giou * cost_giou + self.span_score * cost_class
        C = C.view(B, M, -1).cpu()

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(opt):
    return HungarianMatcher(
        opt.span_score, 
        opt.span_l1, 
        opt.span_giou, 
        opt.span_loss_type,
        opt.max_v_l
    )