CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name base_v1_contras_moco_proj_noReblock

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2