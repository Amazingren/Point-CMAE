CUDA_VISIBLE_DEVICES=5 python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name dev_CL_base_ep300 \
    --ckpts experiments/pretrain/cfgs/debug/ckpt-epoch-300.pth \
    --seed 0