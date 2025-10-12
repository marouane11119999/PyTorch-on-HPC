#!/bin/bash
#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=%u.ddp1node.%j.out
#SBATCH --error=%u.ddp1node.%j.err

module purge
module add slurm
module load gcc/13.2.0
module load openmpi/4.1.6
module load miniconda3
source activate pytorchhpc

export PYTHONWARNINGS="ignore"

# MASTER_ADDR and PORT (single node)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

# Total cores per node
TOTAL_CORES=24

# Loop over number of processes
for PROCS in 2 3 4 6 8; do
    # Calculate cores per process
    CPUS=$((TOTAL_CORES / PROCS))
    echo "Running $PROCS processes on 1 node with $CPUS threads each..."

    # Export thread environment variables
    export OMP_NUM_THREADS=$CPUS
    export MKL_NUM_THREADS=$CPUS
    export NUMEXPR_NUM_THREADS=$CPUS

    # Create unique output folder
    OUTPUT_DIR="_benchmark/1nodes_ddp_${PROCS}proc_${CPUS}threads"
    mkdir -vp ${OUTPUT_DIR}

    mpirun -np $PROCS \
        python3 ../../../src/model_training_benchmark/model_training_ddp_single_node.py \
        --batch_size 128 \
        --num_epochs 5 \
        --num_repeats 1
done
