CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/pretrain_shapenet.yaml \
    --exp_name MocoClusterLinear \
    --val_freq 10