#!/usr/bin/env sh

#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching

#SBATCH --job-name=pytorch_multithread
#SBATCH --output=%u.multithread.%j.out
#SBATCH --error=%u.multithread.%j.err

#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24   # use all cores on one node

# -------------------------
# Load environment
# -------------------------
module purge
module add slurm
module load miniconda3
source activate pytorchhpc

echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is $(pwd)"

# -------------------------
# Output directory
# -------------------------
OUTPUT_DIR="_benchmark/multithread"
mkdir -vp ${OUTPUT_DIR}

# -------------------------
# Run experiments
# -------------------------
# We now test 1, 2, 4, 6, 8, 12, 16, 20, 24 threads
for nthreads in 1 2 4 6 8 12 16 20 24;
do
    export OMP_NUM_THREADS=${nthreads}
    export MKL_NUM_THREADS=${nthreads}

    echo "=========================================="
    echo "Running PyTorch training with ${nthreads} threads"
    echo "=========================================="

    mkdir -vp ${OUTPUT_DIR}/${nthreads}

    srun python3 ../../../src/model_training_benchmark/model_training_multithreading_intraop.py \
        --batch_size 128 \
        --num_epochs 5 \
        --num_repeats 3 \
        &> ${OUTPUT_DIR}/${nthreads}/training_${nthreads}.out
done

source deactivate
echo "Finished at $(date)"
