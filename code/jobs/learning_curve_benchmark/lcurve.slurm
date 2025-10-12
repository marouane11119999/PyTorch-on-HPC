#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=learning_curve
#SBATCH --array=0-5                 # 6 fractions of the dataset
#SBATCH --ntasks=4                  # 4 DDP processes per node
#SBATCH --cpus-per-task=6           # 24 cores / 4 processes = 6 cores per process
#SBATCH --time=01:30:00
#SBATCH --output=%u.learningcurve.%A_%a.out
#SBATCH --error=%u.learningcurve.%A_%a.err

module purge
module add slurm
module load gcc/13.2.0
module load miniconda3
source activate pytorchhpc

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# directory to save results
OUTPUT_DIR="_benchmark/learning_curve"
mkdir -vp ${OUTPUT_DIR}

echo "Running learning curve array task $SLURM_ARRAY_TASK_ID"

srun python3 ../../src/tuning/learning_curve_ddp.py \
    --array_id $SLURM_ARRAY_TASK_ID \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 1
