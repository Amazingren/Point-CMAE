python main.py \
    --root /data/work-gcp-europe-west4-a/bin_ren/point-cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --gpu 0 \
    --ckpts ./log/part_seg/exp_v2-1_ReBlock_MAE0.6_moco_splossv1_noProj_ep250/checkpoints/best_model.pth \
    --seed 1987 \
    --test
