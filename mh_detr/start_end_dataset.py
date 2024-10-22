import logging
import pickle
import random
import sys
from os.path import exists, join

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.basic_utils import l2_normalize_np_array, load_jsonl
from utils.span_utils import span_xx_to_cxw
from utils.tensor_utils import pad_sequences_1d

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dataset, dset_name, data_path, 
                 v_feat_dirs, q_feat_dir, q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0):
        
        self.dataset = dataset
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        
        self.v_feat_dirs = v_feat_dirs if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels in training phase
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

    def load_data(self):
        if self.dataset == "qvhighlights":
            datalist = load_jsonl(self.data_path)
            if self.data_ratio != 1:
                n_examples = int(len(datalist) * self.data_ratio)
                datalist = datalist[:n_examples]
                logger.info("Using {}% of the data: {} examples"
                            .format(self.data_ratio * 100, n_examples))
        elif self.dataset == "charades":
            with open(self.data_path, 'rb') as f:
                datalist = pickle.load(f)
                # "qid": 0, 
                # "query": "a person is putting a book on a shelf.", 
                # "duration": 33.79, 
                # "vid": "AO8RW", 
                # "span": (0.0, 6.9),
                # "query_feat": array.shape([N, 300])
        else:
            raise RuntimeError(f"Dataset: '{self.dataset}' is not defined!")
        return datalist

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if self.dataset == "qvhighlights":
            return self._getitem_qvhighlights(index)
        elif self.dataset == "charades":
            return self._getitem_charades(index)
        else:
            raise RuntimeError(f"Dataset: '{self.dataset}' is not defined!")
    
    def _getitem_qvhighlights(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])       #(N, 512)   N<=32
    
        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])   #(T, 2816)  T<=75
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l
        
        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)          #(T, 2)     start, end
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)   #(T, 2816 + 2)
            else:
                model_inputs["video_feat"] = tef
        
        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)   #(max_windows, 2)
                
            if "subs_train" not in self.data_path:
                model_inputs["saliency_labels"], \
                model_inputs["saliency_pos_labels"], \
                model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_labels"], \
                model_inputs["saliency_pos_labels"], \
                model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)
        
        # print(model_inputs, model_inputs["saliency_pos_labels"],  model_inputs["saliency_neg_labels"])
        
        return dict(meta=meta, model_inputs=model_inputs)
    
    def _getitem_charades(self, index):
        # charades features should be normalized
        
        meta = self.data[index]
        model_inputs = dict()
        
        vid_name = meta["vid"]
        span = meta["span"]
        duration = meta["duration"]
        
        # txt_feat
        if 1:
            q_feat = meta["query_feat"]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
            model_inputs["query_feat"] = torch.from_numpy(q_feat)       #(N, 300)
        
        # vid_feat
        if 1:
            src_v_feat = h5py.File(self.v_feat_dirs[0], 'r')[vid_name][:]
            src_v_l = src_v_feat.shape[0]                           #src_T

            # {span: (14.7, 19.5), duration: 42.19}
            src_start = int(1.0 * src_v_l * span[0] / duration)     #43
            src_end = int(1.0 * src_v_l * span[1] / duration)       #58
            
            if src_end >= src_v_l:
                src_end = src_v_l - 1
            if src_start > src_end:
                src_start = src_end
            assert src_start <= src_end
            assert 0 <= src_start <= src_v_l
            assert 0 <= src_end <= src_v_l
            
            src_span = [src_start, src_end]     #[43, 58]
            
            # size(T, 1024), value[26, 34], np.size(75)
            new_v_feat, new_span, idx = self._sample_charades_vid(src_v_feat, src_span)  
            
            if self.normalize_v:
                new_v_feat = l2_normalize_np_array(new_v_feat)
            model_inputs["video_feat"] = torch.from_numpy(new_v_feat)       #(T, 1024)  T<=75
            new_v_l = new_v_feat.shape[0]         #new_T
        
            # start and end
            tef_st = idx / src_v_l
            tef_ed = np.append(tef_st[1:], 1.0)
            tef = np.stack([tef_st, tef_ed], axis=1)          #(T, 2)     start, end
            model_inputs["video_feat"] = \
                torch.cat([model_inputs["video_feat"], torch.from_numpy(tef)], dim=1)   #(T, 1024 + 2)
        
        # span_label
        if self.load_labels:   
            new_start  = new_span[0] / new_v_l      #0.3467
            new_end =  new_span[1] / new_v_l        #0.4533

            # self.data[index]["relevant_windows"] = [[new_start, new_end]]   #update gt
            new_span = torch.Tensor([new_start, new_end]).unsqueeze(0)      #[0.3467, 0.4533]
            meta["relevant_windows"] = [new_span]
            
            model_inputs["span_labels"] = span_xx_to_cxw(new_span)          #the normalized window and center
            
        return dict(meta=meta, model_inputs=model_inputs)

    def _sample_charades_vid(self, src_v_feat , src_span):
        # sample frame to a fixed num and recompute the label
        
        src_v_l = src_v_feat.shape[0]
        
        # Return evenly spaced numbers over a specified interval. 
        idx = np.linspace(start=0, stop=src_v_l-1, num=self.max_v_l).astype(np.int32) 
        
        new_v_feat = []
        for i in range(len(idx) - 1):
            start = idx[i]
            end = idx[i + 1]
            if start == end or start + 1 == end:
                new_v_feat.append(src_v_feat[start])
            else:
                new_v_feat.append(np.mean(src_v_feat[start: end], 0))
        new_v_feat.append(src_v_feat[-1])
        new_v_feat = np.stack(new_v_feat, 0)
        
        new_span = [0, 0]
        new_span[0] = min(np.where(idx >= src_span[0])[0])
        
        if src_span[1] == src_v_l - 1:
            new_span[1] = self.max_v_l - 1
        else:
            new_span[1] = max(np.where(idx <= src_span[1])[0])
        if new_span[1] < new_span[0]:
            new_span[0] = new_span[1]
            
        return new_v_feat, new_span, idx

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        neg_clip_indices = random.sample(neg_pool, k=max_n)
        
        scores = torch.zeros((75))
        scores[gt_st:gt_ed+1] = 1
        
        return scores, pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1):
        """
        Args:
            rel_clip_ids: list(int), list of relevant clip ids, indices inside rel_clip_ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int, length of video
        Returns:
            scores: [ctx_l, 1]
        """
        scores = np.array(scores)           #[len(rel_clips), 3]
        agg_scores = np.sum(scores, 1)      #[len(rel_clips), ]
        
        scores = torch.zeros((75))
        for i, clip_idx in enumerate(rel_clip_ids):
            scores[clip_idx] = agg_scores[i]
            
        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        sort_indices = np.argsort(agg_scores)  # increasing
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
        if len(easy_neg_pool) >= max_n:
            easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
            easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
        else:  # copy the hard ones
            easy_pos_clip_indices = hard_pos_clip_indices
            easy_neg_clip_indices = hard_neg_clip_indices
        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
            
        return scores, pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (max_windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows
    
    def _get_query_feat_by_qid(self, qid):
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (d, ) or (N, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            _feat_path = join(_feat_dir, f"{vid}.npz")
            _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()   
    #dict_keys(['query_feat', 'video_feat', 'span_labels', 'saliency_labels', 
    #                                       'saliency_pos_labels', 'saliency_neg_labels'])
    
    batched_data = dict()
    
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"][k]) for e in batch]
            continue
        if k == "saliency_labels":
            batched_data[k] = torch.stack([e["model_inputs"][k] for e in batch], dim=0) / 12    #归一化
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        batched_data[k] = pad_sequences_1d([e["model_inputs"][k] for e in batch], 
                                                        dtype=torch.float32, fixed_length=None)
        
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    # model_inputs.keys() == dict_keys(['vid', 'txt', 'vid_mask', 'txt_mask'])
    model_inputs = dict(
        vid = batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        txt = batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        vid_mask = batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        txt_mask = batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
    )
    
    # targets.keys() == dict_keys(['span_labels', 'saliency_labels', 
    #                                   'saliency_pos_labels', 'saliency_neg_labels', 'mask']) 
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [dict(
            spans = e["spans"].to(device, non_blocking=non_blocking)) 
                                    for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_labels" in batched_model_inputs:
        targets["saliency_labels"] = \
                    batched_model_inputs["saliency_labels"].to(device, non_blocking=non_blocking)
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)
        
    targets["mask"] = batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking)
        
    targets = None if len(targets) == 0 else targets
    
    return model_inputs, targets
