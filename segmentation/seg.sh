python main.py \
    --root /workspace/wwang/BINGO/datasets/point_cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --gpu 7 \
    --log_dir ./exp_smconly_mask0 \
    --ckpts ../experiments/base/pretrain/Recon_SMC_Only/ckpt-epoch-300.pth \
    --seed 0