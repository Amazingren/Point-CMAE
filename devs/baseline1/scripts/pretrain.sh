CUDA_VISIBLE_DEVICES=5 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name dev_baseline1

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2