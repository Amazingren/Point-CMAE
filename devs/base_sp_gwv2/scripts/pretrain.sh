CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name base_sp_gwv2

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2