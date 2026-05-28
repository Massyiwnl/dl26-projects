#!/bin/bash
#SBATCH --job-name=multi_seed
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:4096
#SBATCH --time=02:00:00
#SBATCH --output=logs/multi_seed-%j.log

set -e
cd ~/dl26-projects

echo "==========================================================="
echo " Multi-seed run: B1 B2 DANN MMD x seeds (42 123 7) = 12 runs"
echo " Host: $(hostname), Start: $(date)"
echo "==========================================================="

SEEDS=(42 123 7)
SIF=/shared/sifs/latest.sif
FEAT_DIR=data/processed/charades-ego/segment_features

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "############################################"
    echo "###     SEED = $SEED"
    echo "############################################"

    echo "--- [seed=$SEED] B1 source-only ---"
    apptainer exec --nv $SIF python -m src.training.trainer_supervised \
        --train-npz $FEAT_DIR/train_source.npz \
        --val-npz $FEAT_DIR/val_target.npz \
        --output-dir experiments/checkpoints/CLUSTER_b1_seed${SEED} \
        --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 --seed $SEED

    echo "--- [seed=$SEED] B2 target-only oracle ---"
    apptainer exec --nv $SIF python -m src.training.trainer_supervised \
        --train-npz $FEAT_DIR/train_target.npz \
        --val-npz $FEAT_DIR/val_target.npz \
        --output-dir experiments/checkpoints/CLUSTER_b2_seed${SEED} \
        --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 --seed $SEED

    echo "--- [seed=$SEED] DANN lambda=0.5 ---"
    apptainer exec --nv $SIF python -m src.training.trainer_dann \
        --train-source $FEAT_DIR/train_source.npz \
        --train-target $FEAT_DIR/train_target.npz \
        --val-target $FEAT_DIR/val_target.npz \
        --output-dir experiments/checkpoints/CLUSTER_dann_seed${SEED} \
        --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 \
        --lambda-max 0.5 --warmup-epochs 5 --seed $SEED

    echo "--- [seed=$SEED] MMD lambda=1.0 ---"
    apptainer exec --nv $SIF python -m src.training.trainer_mmd \
        --train-source $FEAT_DIR/train_source.npz \
        --train-target $FEAT_DIR/train_target.npz \
        --val-target $FEAT_DIR/val_target.npz \
        --output-dir experiments/checkpoints/CLUSTER_mmd_seed${SEED} \
        --epochs 50 --batch-size 256 --lr 5e-4 --weight-decay 1e-4 \
        --lambda-mmd 1.0 --warmup-epochs 2 --seed $SEED
done

echo ""
echo "==========================================================="
echo " All 12 runs completed at: $(date)"
echo "==========================================================="