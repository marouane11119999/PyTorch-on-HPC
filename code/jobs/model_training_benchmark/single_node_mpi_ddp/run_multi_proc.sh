#!/bin/bash
# Script to submit all 1-node multi-process DDP jobs

echo "Submitting 2-process job..."
dos2unix ddp1node2proc.slurm
sbatch ddp1node2proc.slurm

echo "Submitting 4-process job..."
dos2unix ddp1node4proc.slurm
sbatch ddp1node4proc.slurm

echo "Submitting 8-process job..."
dos2unix ddp1node8proc.slurm
sbatch ddp1node8proc.slurm

echo "All jobs submitted."
