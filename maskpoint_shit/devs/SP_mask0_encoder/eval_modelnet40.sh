CUDA_VISIBLE_DEVICES=5 python main.py \
    --test --deterministic \
    --config cfgs/finetune_modelnet.yaml \
    --ckpts ./experiments/finetune_modelnet/finetune_modelnet40_moco0.05/ckpt-last.pth \
    --exp_name eval_modelnet40_moco0.05_last