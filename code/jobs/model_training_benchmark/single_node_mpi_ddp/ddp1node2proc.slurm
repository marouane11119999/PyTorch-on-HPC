#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=ddp_1node_2proc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # 2 DDP processes
#SBATCH --cpus-per-task=12          # 12 threads per process (2*12=24 cores total)
#SBATCH --time=02:00:00
#SBATCH --output=%u.ddp1node.2proc.%j.out
#SBATCH --error=%u.ddp1node.2proc.%j.err

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

OUTPUT_DIR="_benchmark/1node_ddp_2proc_12threads"
mkdir -vp ${OUTPUT_DIR}

mpirun -np $SLURM_NTASKS_PER_NODE \
    python3 ../../../src/model_training_benchmark/model_training_ddp_single_node.py \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 1
    --output_dir ${OUTPUT_DIR}
