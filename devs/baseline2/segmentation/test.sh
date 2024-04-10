python main.py \
    --root /workspace/wwang/BINGO/datasets/point_cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --gpu 5 \
    --ckpts ./log/part_seg/exp/checkpoints/best_model.pth \
    --seed 5000 \
    --test

# python main.py --gpu $1 --log_dir $2 --ckpts $3 --seed $RANDOM --test