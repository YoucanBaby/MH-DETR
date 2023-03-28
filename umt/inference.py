import logging
import os
import pprint
import sys
import time
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

sys.path.append("/home/xuyifang/VGHD/Moment-DETR")

from standalone_eval.eval import eval_submission
from umt.config import TestOptions
from umt.models.loss import build_criterion
from umt.models.umt import build_umt, build_umt_v2, build_umt_v3, build_umt_v4
from umt.postprocessing import PostProcessor
from umt.start_end_dataset import (StartEndDataset, prepare_batch_inputs,
                                   start_end_collate)
from utils.basic_utils import AverageMeter, save_json, save_jsonl
from utils.span_utils import span_cxw_to_xx
from utils.temporal_nms import temporal_nms

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in "val":  # since test_public has no GT
        
        metrics = eval_submission(
            submission, gt_data,
            opt,
            verbose=opt.debug, match_number=not opt.debug
        )
        
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
        
    return metrics, metrics_nms, latest_file_paths


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        outputs = model(**model_inputs)
        prob = F.softmax(outputs["score"], -1)  #(B, #queries, #classes=2)
        
        if opt.span_loss_type == "l1":
            scores = prob[..., 0]           #(B, #queries)  foreground label is 0, we directly take it
            pred_spans = outputs["span"]    #(B, #queries, 2)
            
            _saliency_scores = outputs["saliency"].half()  # (B, T)
            saliency_scores = []
            valid_vid_lengths = model_inputs["vid_mask"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            # TODO use more advanced decoding method with st_ed product
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            if opt.span_loss_type == "l1":
                if opt.dataset == "qvhighlights":
                    spans = span_cxw_to_xx(spans) * meta["duration"]
                else:
                    spans = span_cxw_to_xx(spans) * opt.max_v_l * opt.clip_length
                
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            if not opt.no_sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx]
            )
            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.debug:
            break

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    post_processor = PostProcessor(
        clip_length=2, min_ts_val=0, max_ts_val=150,
        min_w_l=2, max_w_l=150, move_window_method="left",
        process_func_names=("clip_ts", "round_multiple")
    )
    mr_res = post_processor(mr_res)
    
    return mr_res, loss_meters


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    
    if opt.no_sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
        
    metrics, metrics_nms, latest_file_paths = \
                    eval_epoch_post_processing(submission, opt, eval_dataset.data, save_submission_filename)
    
    # if visualize res, please set "opt.draw_res=True".
    opt.draw_res = True
    if opt.draw_res:
        vis_dir = "/home/xuyifang/VGHD/Moment-DETR/visualization/"
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir) 
        
        qid_list = [
            2916, 4590
        ]
        stop = 60 
        pred_sub = 0.15
                
        for pred, data in tqdm(zip(submission, eval_dataset.data), desc="Save visualization results"):
            qid = data["qid"]
            if qid not in qid_list:
                continue
            
            fig, ax = plt.subplots(figsize=(32, 2))

            x = np.arange(0, 75, 1) * 2
            pred_hl, data_hl = np.zeros(75), np.zeros(75)
            
            pred_rel_hl = np.array(pred["pred_saliency_scores"])
            data_rel_hl = np.sum(data["saliency_scores"], axis=1) / 12
            
            for idx, rel_idx in enumerate(data["relevant_clip_ids"]):
                data_hl[rel_idx] = data_rel_hl[idx]
                pred_hl[rel_idx] = pred_rel_hl[idx]
            
            pred_hl -= pred_sub
            pred_hl = np.round(np.clip(pred_hl, 0, 1), 2)
            data_hl = np.round(data_hl, 2)
            
            if stop > 75:
                ax.plot(x, data_hl, label='Ground Truth', linestyle='-', color='lightcoral', linewidth=3)
                ax.plot(x, pred_hl, label='Prediction', linestyle='--', color='darkgreen', linewidth=3)
            else:
                ax.plot(x[:stop], data_hl[:stop], label='Ground Truth (Query1)', linestyle='-', color='lightcoral', linewidth=5)
                ax.plot(x[:stop], pred_hl[:stop], label='Prediction (Query1)', linestyle='--', color='forestgreen', linewidth=5)
                
                # ax.plot(x[:stop], data_hl[:stop], label='Ground Truth (Query2)', linestyle='-', color='palevioletred', linewidth=5)
                # ax.plot(x[:stop], pred_hl[:stop], label='Prediction (Query2)', linestyle='--', color='teal', linewidth=5)
                
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)
            
            ax.xaxis.set_ticks(np.arange(0, 150, 30))
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.5))
            
            ax.set_xmargin(0)
        
            ax.legend(fontsize=18)
            fig.savefig(f"{vis_dir}/{qid}_label.png", dpi=150)
            
            # ax.legend().set_visible(False)      #label不可见
            # fig.savefig(f"{vis_dir}/{qid}.png", dpi=150)
            
            plt.close()

    
    # 获得一个vid下的多个qid
    if 0:
        vid_dict = {}
        res_dict = {}
        
        for data in tqdm(eval_dataset.data):
            qid = data["qid"]
            vid = data["vid"]
            
            if vid not in vid_dict:
                vid_dict[vid] = set()
            vid_dict[vid].add(qid)
            
            if len(vid_dict[vid]) >= 2:
                res_dict[vid] = vid_dict[vid]
        
        for item in res_dict.items():
            print(item)
        

    return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/criterion/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/criterion/optimizer/scheduler")
    
    # model = build_umt(opt)
    model = build_umt_v4(opt)
    criterion = build_criterion(opt)
    
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    # https://huggingface.co/docs/timm/reference/optimizers
    optimizer = create_optimizer(opt, model)
    lr_scheduler, _ = create_scheduler(opt, optimizer)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    eval_dataset = StartEndDataset(
        dataset=opt.dataset,
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True if opt.eval_split_name == "val" else False,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0
    )

    model, criterion, _, _ = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_preds.jsonl".format(
                                            opt.dset_name, opt.eval_split_name, opt.eval_id)
    logger.info("Starting inference...")
    t = time.time()
    
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=criterion)
    
    logger.info(f"Inference cost: {(time.time() - t):.3f}s")
    logger.info(f"Inference QPS: {round(1550 / (time.time() - t))}") 
    
    if metrics_no_nms is not None:
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))


if __name__ == '__main__':
    start_inference()
