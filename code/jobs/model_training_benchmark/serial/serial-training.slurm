#!/usr/bin/env sh

#SBATCH --reservation=hpc-course-sose2025
#SBATCH --partition=teaching

#SBATCH --job-name=baseline_training
#SBATCH --output=baseline_training_%j.out
#SBATCH --error=baseline_training_%j.err

#SBATCH --time=0-02:00:00         
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1         # serial training

# -------------------------
# Load environment
# -------------------------
module purge
module add slurm
module load miniconda3
source activate pytorchhpc

# Test imports
python -c "import numpy; import torch; print('Imports OK')"

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# -------------------------
# Output directory
# -------------------------
OUTPUT_DIR="_benchmark/baseline"
mkdir -vp ${OUTPUT_DIR}

# -------------------------
# Run baseline training
# -------------------------
srun python3 ../src/benchmarks/baseline_training_benchmark.py \
    --batch_size 128 \
    --epochs 5 \
    --num_repeats 3 \
    &> ${OUTPUT_DIR}/baseline_training.out

echo "Finished at `date`"
