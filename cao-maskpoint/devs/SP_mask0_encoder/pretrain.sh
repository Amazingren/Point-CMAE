CUDA_VISIBLE_DEVICES=4 python main.py \
    --config cfgs/pretrain_shapenet.yaml \
    --exp_name SP_mask0 \
    --resume \
    --val_freq 10