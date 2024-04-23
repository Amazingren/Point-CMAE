CUDA_VISIBLE_DEVICES=5 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name dev_sponly_noNorm

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2