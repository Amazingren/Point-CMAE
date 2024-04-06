CUDA_VISIBLE_DEVICES=7 python main.py \
    --test \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --exp_name scan_hardest_reproduce \
    --ckpts experiments/finetune_scan_hardest/full/scan_hardest_reproduce/ckpt-best.pth \
    --seed 0 

# CUDA_VISIBLE_DEVICES=$1 python main.py --test --config cfgs/full/finetune_modelnet.yaml --exp_name $2 --ckpts $3 --seed $RANDOM 
