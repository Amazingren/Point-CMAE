#!/bin/bash
#SBATCH --job-name=dev_DM_cross_cos_1_seg_ep275
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/dev_DM_cross_cos_1_seg_ep275.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/dev_DM_cross_cos_1_seg_ep275.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate points

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/dev_DM_cross_cos_1/segmentation

python main.py \
    --root /data/work-gcp-europe-west4-a/bin_ren/point-cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --log_dir ./dev_DM_cross_cos_1_seg_ep275 \
    --ckpts ../experiments/pretrain/cfgs/dev_DM_cross_cos_mask1/ckpt-epoch-275.pth