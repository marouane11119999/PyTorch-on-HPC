#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=hyperparam_search
#SBATCH --array=0-8                      # 9 combinations (3 lrs × 3 batch_sizes)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%u.hyperparam.%A_%a.out
#SBATCH --error=%u.hyperparam.%A_%a.err

module purge
module add slurm
module load gcc/13.2.0
module load miniconda3
source activate pytorchhpc

echo "Running array job $SLURM_ARRAY_TASK_ID"
echo "============================================"
echo " SLURM JOB INFO "
echo " Job ID: $SLURM_JOB_ID"
echo " Array Task ID: $SLURM_ARRAY_TASK_ID"
echo " Node: $(hostname)"
echo " Start Time: $(date)"
echo "============================================"



python3 ../../src/tuning/hyperparameter_search.py \
    --array_id $SLURM_ARRAY_TASK_ID \
    --output_dir "_benchmark/hyperparam_search"
