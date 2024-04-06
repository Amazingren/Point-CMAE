CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_reproduce \
    --ckpts experiments/base/pretrain/Recon_SM_reproduce/ckpt-epoch-300.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 