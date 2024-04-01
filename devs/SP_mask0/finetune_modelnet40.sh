CUDA_VISIBLE_DEVICES=3 python main.py \
    --config cfgs/finetune_modelnet.yaml \
    --finetune_model \
    --ckpts ./experiments/pretrain_shapenet/reproduce_maskPoint_moco0.05/ckpt-last.pth \
    --exp_name finetune_modelnet40_moco0.05