CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/finetune_modelnet.yaml \
    --finetune_model \
    --ckpts ./experiments/pretrain_shapenet/maskPoint_cluster/ckpt-last.pth \
    --exp_name finetune_modelnet40_cluster