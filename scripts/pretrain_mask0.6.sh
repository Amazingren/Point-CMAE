CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/pretrain/base_mask0.6.yaml \
    --exp_name Recon_SMC_Only_mask0.6 \
    --resume

# Final Command
# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/pretrain/base.yaml --exp_name $2