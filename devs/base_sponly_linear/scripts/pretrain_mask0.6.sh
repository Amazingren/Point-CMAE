CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/pretrain/base_mask0.6.yaml \
    --exp_name dev_sponly_m0.6

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2