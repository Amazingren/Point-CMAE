CUDA_VISIBLE_DEVICES=4 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name base_v0_contras_dino_noReBlock

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2