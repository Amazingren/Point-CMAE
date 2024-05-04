CUDA_VISIBLE_DEVICES=5 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name base_v1_contras-dino_proj_noReblock

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2