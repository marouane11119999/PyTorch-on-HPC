#!/bin/bash
#SBATCH --job-name=pytorch_multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --mem=8G

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
OUTPUT_DIR="_benchmark/multithreading"
mkdir -vp ${OUTPUT_DIR}

python multicore_benchmark.py --batch_size 128 --num_epochs 3 --num_repeats 3
