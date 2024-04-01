CUDA_VISIBLE_DEVICES=7 python main.py \
    --config cfgs/fewshot_modelnet.yaml \
    --finetune_model \
    --ckpts ./experiments/pretrain_shapenet/reproduce_maskPoint/ckpt-last.pth \
    --exp_name finetune_fewshot_modelnet \
    --way 5 \
    --shot 10 \
    --fold 0