CUDA_VISIBLE_DEVICES=5 python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name dev_CL_base_cls_ep100 \
    --ckpts experiments/pretrain/cfgs/debug/ckpt-epoch-100.pth \
    --seed 0