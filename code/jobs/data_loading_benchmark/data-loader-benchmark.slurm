#!/usr/bin/env sh

#SBATCH --reservation=hpc-course-sose2025 
#SBATCH --partition=teaching
 
#SBATCH --job-name=data_loader_benchmark
#SBATCH --output=data_loader_benchmark_%j.out
#SBATCH --error=data_loader_benchmark_%j.err

#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24


module purge # clear current environment
### load modules needed by your app
module add slurm # you will always need this one
module load miniconda3
source activate pytorchhpc
python -c "import numpy; import torch; print('Imports OK')"
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`" 


# Set output directory for benchmark results
OUTPUT_DIR="_benchmark/data_loading"
mkdir -vp ${OUTPUT_DIR}

srun python3 ../../src/data_loading_benchmark/data_loading_benchmark.py \
    --batch_size 128 \
    --num_repeats 5 \
    --workers_list 1 2 4 8 12 24 \
    &> ${OUTPUT_DIR}/benchmark_all_workers.out

echo "Finished at `date`"
