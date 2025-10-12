#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=learning_curve_ddp
#SBATCH --array=0-5                    # 6 fractions for the learning curve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2            # 2 DDP processes per job
#SBATCH --cpus-per-task=12             # 12 threads per process (2*12=24 cores total)
#SBATCH --time=02:00:00
#SBATCH --output=%u.learning_curve.%A_%a.out
#SBATCH --error=%u.learning_curve.%A_%a.err

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

# Create an output folder specific to this array task
OUTPUT_DIR="_benchmark/learning_curve/frac_${SLURM_ARRAY_TASK_ID}_ddp2proc_12threads"
mkdir -vp ${OUTPUT_DIR}

echo "Launching learning curve job for fraction index $SLURM_ARRAY_TASK_ID on node $(hostname)"
echo "Output will be saved to $OUTPUT_DIR"

# Launch DDP processes using MPI
mpirun -np $SLURM_NTASKS_PER_NODE \
    python3 ../../../src/learning_curve/learning_curve_ddp.py \
    --array_id $SLURM_ARRAY_TASK_ID \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 1
