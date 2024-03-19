CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/pretrain_shapenet.yaml \
    --exp_name cluster_dev \
    --val_freq 10