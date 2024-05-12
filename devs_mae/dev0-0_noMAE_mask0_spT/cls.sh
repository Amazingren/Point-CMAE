CUDA_VISIBLE_DEVICES=4 python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name dev0-0_noMAE_mask0_spT_ep100 \
    --ckpts experiments/pretrain/cfgs/dev_sp_noMAE_mask0/ckpt-epoch-100.pth \
    --seed 0