import argparse
import os
import time

import torch

from utils.basic_utils import (dict_to_markdown, load_json, make_zipfile,
                               mkdirp, save_json)


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser('Model training and evaluation script')
        
        parser.add_argument("--train_batch_size", type=int, default=32, help="mini-batch size at training")
        parser.add_argument("--eval_batch_size", type=int, default=100, help="mini-batch size at inference, for query")
        parser.add_argument("--epochs", type=int, default=200, help="number of epochs to run")
        parser.add_argument("--eval_epoch_interval", type=int, default=5, help="number of interval eval at training")
        
        parser.add_argument("--dset_name", type=str, default="umt")
        parser.add_argument("--eval_split_name", type=str, default="val", 
                            help="should match keys in video_duration_idx_path, must set for MR")
        parser.add_argument("--draw_res", action="store_true", help="Save visualization res for MR and HD")
        parser.add_argument("--debug", action="store_true",
                            help="debug (fast) mode, break all loops, do not load all data into memory.")
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default="exp0", help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2018, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # Dataset config
        parser.add_argument("--dataset", type=str, default="qvhighlights", choices=['qvhighlights', 'charades'])
        parser.add_argument("--data_ratio", type=float, default=1.0,
                    help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                            "Use small portion for debug purposes. Note this is different from --debug, "
                            "which works by breaking the loops, typically they are not used together.")
        parser.add_argument("--max_v_l", type=int, default=75)
        parser.add_argument("--max_q_l", type=int, default=32)
        parser.add_argument("--clip_length", type=int, default=2)
        parser.add_argument("--max_windows", type=int, default=5)
        parser.add_argument("--train_path", type=str, default="data/highlight_train_release.jsonl")
        parser.add_argument("--eval_path", type=str, default="data/highlight_val_release.jsonl",
                            help="Evaluating during training, for Dev set. If None, will only do training")
        parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
        parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
        parser.add_argument("--v_feat_dirs", type=str, nargs="+",
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here." 
                                 "Dirs are pre-defined in shell script.")
        parser.add_argument("--t_feat_dir", type=str, default="features/clip_text_features/", help="text/query feature dir")
        parser.add_argument("--v_feat_dim", type=int, default=2816, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, default=512, help="text/query feature dim")
        parser.add_argument("--ctx_mode", type=str, default="video_tef", help="use video feat and text feat")
        parser.add_argument("--text_drop_ratio", type=float, default=0, help="drop text_drop_ratio tokens from text input. 0.1=10%")  
        
        # Training config
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")
        
        # Optimizer hyper-parameters        
        parser.add_argument('--opt', type=str, default='adamw', metavar='OPTIMIZER', choices=['adamw', 'adam', 'sgd'], 
                            help='Optimizer (default: "adamw"')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')
        parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        
        # Learning rate schedule hyper-parameters
        parser.add_argument('--sched', type=str, default='step', metavar='SCHEDULER', choices=['cosine', 'step'], 
                            help='LR scheduler (default: "step")')
        parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
        parser.add_argument('--warmup-lr', type=float, default=5e-7, metavar='LR',
                        help='warmup learning rate (default: 5e-7)')
        parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
        parser.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
        parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                            help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
        parser.add_argument('--decay-epochs', type=float, default=1000, metavar='N',
                            help='epoch interval to decay LR')
        parser.add_argument('--decay-rate', '--dr', type=float, default=0.8, metavar='RATE',
                            help='LR decay rate (default: 0.1)')

        # Model config
        parser.add_argument('--activation', type=str, default="relu", choices=["relu", "gelu", "glu"], help="Activation function of the model")
        ## * Input FNN 
        parser.add_argument("--num_input_ffn_layers", type=int, default=2, help="#layers to encoder input")
        parser.add_argument('--input_vid_ffn_dropout', type=float, default=0.5, help="Dropout applied in input video ffn")
        parser.add_argument('--input_txt_ffn_dropout', type=float, default=0.3, help="Dropout applied in input text ffn")
        ## * Transformer
        parser.add_argument('--qkv_dim', type=int, default=256, help="Dimension of the transformer")
        parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_vg_qry', type=int, default=10, help="Number of video grounding queries")
        parser.add_argument('--dropout', type=float, default=0.1, help="Dropout applied in the transformer and output ffn")
        parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate (default: 0.1)')
        
        # * Loss weight
        parser.add_argument("--saliency_bce", type=float, default=1, help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument('--saliency_hinge', type=float, default=0.1)
        parser.add_argument('--span_align', type=float, default=0)
        parser.add_argument('--span_score', type=float, default=4, help="VG score weight for the matching cost and loss")
        parser.add_argument('--span_l1', type=float, default=10, help="VG L1 weight for the matching cost and loss")
        parser.add_argument('--span_giou', type=float, default=1, help="VG gIoU weight for the matching cost and loss")
        parser.add_argument("--coarse_contrastive", type=float, default=0, help="loss weight of coarse-grained contrastive learning")
        parser.add_argument("--vid_qry_contrastive", type=float, default=0, help="loss weight of contrastive learning between vid and qry")
        ## * Loss hyper-parameters
        parser.add_argument('--coef_eos', type=float, default=0.1, help="Relative classification coefficient of the no-object class")
        parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss, do not change")
        
        
        #TODO delete
        if 1:
            # Loss and model config
            ## Contrastive learning loss
            parser.add_argument("--contrastive_align_loss", action="store_true", help="Enable contrastive_align_loss between matched query spans and the text.")
            parser.add_argument("--contrastive_dim", type=int, default=64, help="dim for contrastive embeddings")
            ## others
            parser.add_argument('--aux_loss', action='store_true', help="Enable auxiliary decoding losses (loss at each layer)")
            parser.add_argument("--span_loss_type", default="l1", choices=['l1', 'ce'], type=str,
                                help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
              
        #TODO delete
        if 1:
            # Do not change
            # Ambiguous hyper-pparameters
            parser.add_argument("--max_before_nms", type=int, default=10)
            parser.add_argument("--max_after_nms", type=int, default=10)
            parser.add_argument("--no_sort_results", action="store_true", help="do not sort results, use this for moment query visualization")
            parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
            parser.add_argument("--nms_thd", type=float, default=-1,
                                help="additionally use non-maximum suppression "
                                    "(or non-minimum suppression for distance)"
                                    "to post-processing the predictions. "
                                    "-1: do not use nms. [0, 1]")
            parser.add_argument("--saliency_margin", type=float, default=0.2)
        
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()

        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.num_workers = 0

        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            # opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            opt.model_dir = os.path.dirname(opt.resume)
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            # use saved options to overwrite all BaseOptions args.
            for arg in saved_options:  
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",  # "max_before_nms", "max_after_nms"
                               "max_pred_l", "min_pred_l",
                               "resume", "resume_all", "no_sort_results", 
                               "eval_split_name", "eval_path"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
            if opt.eval_results_dir is not None:
                opt.results_dir = opt.eval_results_dir
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            ctx_str = opt.ctx_mode + "_sub" if any(["sub_ctx" in p for p in opt.v_feat_dirs]) else opt.ctx_mode
            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([opt.dset_name, ctx_str, opt.exp_id,
                                                     time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            # save a copy of current code
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        self.display_save(opt)

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        opt.use_tef = "tef" in opt.ctx_mode
        opt.use_video = "video" in opt.ctx_mode
        if not opt.use_video:
            opt.v_feat_dim = 0
        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--eval_results_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
