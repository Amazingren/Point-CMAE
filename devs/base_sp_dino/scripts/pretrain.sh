CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name base_sp_dino

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2