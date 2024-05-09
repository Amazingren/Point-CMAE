python main.py \
    --root /workspace/wwang/BINGO/datasets/point_cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --gpu 5 \
    --log_dir ./exp_MAEonly_mask0.6 \
    --ckpts ../experiments/base/pretrain/Recon_MAE0.6_Only/ckpt-epoch-300.pth \
    --seed 0