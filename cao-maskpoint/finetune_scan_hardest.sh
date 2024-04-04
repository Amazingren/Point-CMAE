CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/finetune_scanobject_hardest.yaml \
    --finetune_model \
    --ckpts ./experiments/pretrain_shapenet/maskPoint_cluster/ckpt-last.pth \
    --exp_name finetune_scan_hardest_cluster