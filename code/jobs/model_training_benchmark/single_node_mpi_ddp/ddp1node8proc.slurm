#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=ddp_1node_8proc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8         # 8 DDP processes
#SBATCH --cpus-per-task=3           # 3 threads per process (8*3=24 cores)
#SBATCH --time=02:00:00
#SBATCH --output=%u.ddp1node.8proc.%j.out
#SBATCH --error=%u.ddp1node.8proc.%j.err

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

OUTPUT_DIR="_benchmark/1node_ddp_8proc_3threads"
mkdir -vp ${OUTPUT_DIR}

mpirun -np $SLURM_NTASKS_PER_NODE \
    python3 ../../../src/model_training_benchmark/model_training_ddp_single_node.py \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 1 \
    --output_dir ${OUTPUT_DIR}
