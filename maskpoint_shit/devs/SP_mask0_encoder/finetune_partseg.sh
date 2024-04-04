cd segmentation
CUDA_VISIBLE_DEVICES=7 python main.py \
    --ckpts ../experiments/pretrain_shapenet/SP_mask0/ckpt-epoch-300.pth \
    --root /workspace/wwang/BINGO/datasets/point_cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --learning_rate 0.0002 \
    --epoch 300 \
    --gpu 7