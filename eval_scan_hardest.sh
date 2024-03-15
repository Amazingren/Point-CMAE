CUDA_VISIBLE_DEVICES=7 python main.py \
    --test --deterministic \
    --config cfgs/finetune_scanobject_hardest.yaml \
    --ckpts ./experiments/finetune_scanobject_hardest/finetune_scan_hardest_cluster/ckpt-last.pth \
    --exp_name eval_scan_hardest_cluster_last