CUDA_VISIBLE_DEVICES=2 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_sponly_mask0.6_ep250 \
    --ckpts experiments/base_mask0.6/pretrain/dev_sponly_m0.6/ckpt-epoch-250.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 