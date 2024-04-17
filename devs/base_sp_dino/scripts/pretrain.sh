CUDA_VISIBLE_DEVICES=5 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name base_sp_dino_dev

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2