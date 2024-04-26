CUDA_VISIBLE_DEVICES=6 python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name scan_hardest_sp_linear_student_ep300 \
    --ckpts experiments/base/pretrain/base_sp_linear_student/ckpt-epoch-300.pth \
    --seed 0

# CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
# --finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 