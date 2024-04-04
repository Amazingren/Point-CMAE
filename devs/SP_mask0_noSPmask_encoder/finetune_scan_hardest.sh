CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/finetune_scanobject_hardest.yaml \
    --finetune_model \
    --ckpts ./experiments/pretrain_shapenet/SP_mask0/ckpt-epoch-300.pth \
    --exp_name finetune_scan_hardest_SP_mask0_ep300
