CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_MAEonly_mask0.6_ep250 \
    --ckpts experiments/base/pretrain/Recon_MAE0.6_Only/ckpt-epoch-250.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 