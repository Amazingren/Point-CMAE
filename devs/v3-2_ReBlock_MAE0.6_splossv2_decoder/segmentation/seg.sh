python main.py \
    --root /workspace/wwang/BINGO/datasets/point_cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --gpu 7 \
    --log_dir ./exp_v3-2_ReBlock_MAE0.6_splossv2_decoder_ep150 \
    --ckpts ../experiments/base/pretrain/v3-1_ReBlock_MAE0.3_splossv2_decoder/ckpt-epoch-150.pth \
    --seed 0