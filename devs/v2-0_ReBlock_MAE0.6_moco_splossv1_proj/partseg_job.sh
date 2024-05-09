#!/bin/bash
#SBATCH --job-name=v2-0_seg
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/v2-0_seg_ep300.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/v2-0_seg_ep300.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs/v2-0_ReBlock_MAE0.6_moco_splossv1_proj/segmentation

python main.py \
    --root /data/work-gcp-europe-west4-a/bin_ren/point-cloud/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --log_dir ./exp_v2-0_ReBlock_MAE0.6_moco_splossv1_proj_ep300 \
    --ckpts ../experiments/base/pretrain/v2-0_ReBlock_MAE0.6_moco_splossv1_proj/ckpt-epoch-300.pth \
    --seed 0