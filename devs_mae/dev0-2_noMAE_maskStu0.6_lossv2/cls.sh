CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name dev0-2_noMAE_maskStu0.6_lossv2 \
    --ckpts experiments/pretrain/cfgs/dev0-2_noMAE_maskStu0.6_lossv2/ckpt-epoch-300.pth \
    --seed 0