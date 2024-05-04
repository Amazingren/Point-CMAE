CUDA_VISIBLE_DEVICES=4 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_v0_contras-moco_noReBlock_ep300 \
    --ckpts experiments/base/pretrain/base_v0_contras_moco_noReBlock/ckpt-epoch-300.pth\
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 