CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name mae_sp_v1_ep100 \
    --ckpts experiments/pretrain/cfgs/dev_sp_v1/ckpt-epoch-100.pth \
    --seed 0