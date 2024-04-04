CUDA_VISIBLE_DEVICES=6 python main.py \
    --test --deterministic \
    --config cfgs/finetune_modelnet.yaml \
    --ckpts ./experiments/finetune_modelnet/finetune_modelnet40_cluster/ckpt-last.pth \
    --exp_name eval_modelnet40_last