#!/bin/bash


#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching

#SBATCH --job-name=ddp_2nodes
#SBATCH --nodes=2                   # total 2 nodes
#SBATCH --ntasks-per-node=1         # 1 process per node (1 DDP rank)
#SBATCH --cpus-per-task=24          # 24 cores per process
#SBATCH --time=01:00:00
#SBATCH --output=%u.ddp2nodes.%j.out
#SBATCH --error=%u.ddp2nodes.%j.err

# ================================
# Load necessary modules
# ================================
module purge
module add slurm
module load miniconda3
module add openmpi/4.1.5/gcc/11.3.0

source activate pytorchhpc


# ================================
# Environment setup for DDP
# ================================
# Number of threads per rank
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONWARNINGS="ignore"

# PyTorch DDP variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# ================================
# Logging info
# ================================
echo "=========================================="
echo " Running PyTorch DDP on 2 Nodes"
echo " SLURM Job ID: ${SLURM_JOB_ID}"
echo " Node list: ${SLURM_NODELIST}"
echo " Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo " CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo " MASTER_ADDR: ${MASTER_ADDR}"
echo " WORLD_SIZE: ${WORLD_SIZE}"
echo "=========================================="
echo


OUTPUT_DIR="_benchmark/2nodes_ddp"
mkdir -vp ${OUTPUT_DIR}
# ================================
# Run your benchmark
# ================================
srun python3 ../../../src/model_training_benchmark/model_training_ddp.py \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 3

echo
echo "Job finished at $(date)"
