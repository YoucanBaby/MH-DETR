ckpt_path=$1
eval_split_name=$2
eval_path=data/charades/val.pkl

PYTHONPATH=$PYTHONPATH:. python umt/inference.py \
--dataset charades \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path}
${@:3}
