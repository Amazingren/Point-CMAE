CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_sponly_mask0.6_linearboth_ep275 \
    --ckpts experiments/base/pretrain/dev_sponly_linearboth/ckpt-epoch-275.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 