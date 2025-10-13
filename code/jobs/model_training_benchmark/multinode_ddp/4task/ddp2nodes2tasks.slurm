#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --job-name=ddp_2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=%u.ddp2nodes4tasks.%j.out
#SBATCH --error=%u.ddp2nodes4tasks.%j.err

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

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=12355

OUTPUT_DIR="_benchmark/2nodes_ddp_4task"
mkdir -vp ${OUTPUT_DIR}

HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print $1":4"}' | paste -sd "," -)

#HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd "," -)
echo "Launching on hosts: $HOSTS"

mpirun -np $SLURM_NTASKS --host $HOSTS \
    python3 ../../../../src/model_training_benchmark/model_training_ddp.py \
    --batch_size 128 \
    --num_epochs 5 \
    --num_repeats 1
