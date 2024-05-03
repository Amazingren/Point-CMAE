python main.py \
    --root /workspace/wwang/BINGO/datasets/point_cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --gpu 6 \
    --ckpts ../experiments/base/pretrain/base_v0_contras_moco/ckpt-epoch-250.pth \
    --log_dir ./exp_v0_contras_moco_ep250 \
    --seed 0