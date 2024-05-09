CUDA_VISIBLE_DEVICES=5 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name v3-2_ReBlock_MAE0.6_splossv2_decoder_ep150 \
    --ckpts experiments/base/pretrain/v3-1_ReBlock_MAE0.3_splossv2_decoder/ckpt-epoch-150.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 