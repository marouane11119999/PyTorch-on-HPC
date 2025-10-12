#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=validation_curve_ddp
#SBATCH --array=0-3                    # 4 parameter values per experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2            # 2 DDP processes per job
#SBATCH --cpus-per-task=12             # 12 threads per process
#SBATCH --time=02:00:00
#SBATCH --output=%u.validation_curve.%A_%a.out
#SBATCH --error=%u.validation_curve.%A_%a.err

# -----------------------------
# Environment setup
# -----------------------------
module purge
module add slurm
module load gcc/13.2.0
module load openmpi/4.1.6
module load miniconda3
source activate pytorchhpc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONWARNINGS="ignore"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

# -----------------------------
# Read parameter type from environment variable
# -----------------------------
if [ -z "$ARG_TYPE" ]; then
    echo "ARG_TYPE not set! Exiting..."
    exit 1
fi

echo "Launching validation curve job for ${ARG_TYPE} index $SLURM_ARRAY_TASK_ID on node $(hostname)"

# -----------------------------
# Output directory setup
# -----------------------------
OUTPUT_DIR="_benchmark/validation_curve/${ARG_TYPE}_ddp2proc_${SLURM_CPUS_PER_TASK}threads"
mkdir -vp ${OUTPUT_DIR}

# -----------------------------
# Launch DDP Python script
# -----------------------------
mpirun -np $SLURM_NTASKS_PER_NODE \
    python3 ../../src/validation_curve/validation_curve.py \
    --array_id $SLURM_ARRAY_TASK_ID \
    --output_dir ${OUTPUT_DIR} \
    --arg_type $ARG_TYPE \
    --num_repeats 1
