#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=ddp_1node_multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # 4 processes on one node
#SBATCH --cpus-per-task=6           # 24 cores / 4 processes = 6 cores per process
#SBATCH --time=02:00:00
#SBATCH --output=%u.ddp1node.%j.out
#SBATCH --error=%u.ddp1node.%j.err

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

# MASTER_ADDR and PORT (since single node, use localhost)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

OUTPUT_DIR="_benchmark/1node_ddp"
mkdir -vp ${OUTPUT_DIR}

echo "Running ${SLURM_NTASKS_PER_NODE} processes on a single node..."

mpirun -np $SLURM_NTASKS_PER_NODE \
    python3 ../../../src/model_training_benchmark/model_training_ddp_single_node.py \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 1
