CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_sp_linear_both_ep275 \
    --ckpts experiments/base/pretrain/base_sp_linear_both/ckpt-epoch-275.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 