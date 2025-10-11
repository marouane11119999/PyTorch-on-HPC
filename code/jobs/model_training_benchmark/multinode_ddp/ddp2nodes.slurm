#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=ddp_2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=%u.ddp2nodes.%j.out
#SBATCH --error=%u.ddp2nodes.%j.err

module purge
module load gcc/13.2.0
module load openmpi/4.1.6
module load miniconda3
source activate pytorchhpc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONWARNINGS="ignore"

OUTPUT_DIR="_benchmark/2nodes_ddp"
mkdir -vp ${OUTPUT_DIR}

# Expand Slurm node list for OpenMPI
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd "," -)
echo "Launching on hosts: $HOSTS"

mpirun -np $SLURM_NTASKS --host $HOSTS \
    python3 ../../../src/model_training_benchmark/model_training_ddp.py \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 3
