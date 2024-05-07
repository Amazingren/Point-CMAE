CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_SMConly_mask0.6_ep300 \
    --ckpts experiments/base_mask0.6/pretrain/Recon_SMC_Only_mask0.6/ckpt-epoch-300.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 