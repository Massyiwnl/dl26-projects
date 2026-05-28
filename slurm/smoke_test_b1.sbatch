#!/bin/bash
#SBATCH --job-name=smoke_b1
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:4096
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke_b1-%j.log

set -e
cd ~/dl26-projects

echo "=== Smoke test: B1 source-only, 5 epochs ==="
echo "Host: $(hostname), Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv /shared/sifs/latest.sif python -m src.training.trainer_supervised \
    --train-npz data/processed/charades-ego/segment_features/train_source.npz \
    --val-npz data/processed/charades-ego/segment_features/val_target.npz \
    --output-dir experiments/checkpoints/CLUSTER_smoke_b1 \
    --epochs 5 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 --seed 42

echo "=== Smoke test completed ==="