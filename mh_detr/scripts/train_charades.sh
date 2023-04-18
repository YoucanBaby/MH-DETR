# training
PYTHONPATH=$PYTHONPATH:. python mh_detr/train.py \
--dset_name charades-utr-cheat_0.05 \
--dataset charades \
--train_path data/charades/train_val_0.05.pkl \
--eval_path data/charades/val.pkl \
--v_feat_dirs features/charades_features/i3d.hdf5 \
--t_feat_dir data/charades \
--v_feat_dim 1024 \
--t_feat_dim 300 \
--max_v_l 75 \
--max_q_l 10 \
--saliency_bce 0 \
--saliency_hinge 0 \
--eval_epoch_interval 1 \
--epochs 200 \
--lr 2e-4 \
--train_batch_size 32 
