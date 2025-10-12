#!/bin/bash

# Loop over all parameter types
for param in learning_rate batch_size optimizer num_epochs; do
    echo "Submitting validation curve jobs for parameter: $param"
    sbatch \
        --export=ARG_TYPE=$param \
        --output=%u.validation_curve_${param}.%A_%a.out \
        --error=%u.validation_curve_${param}.%A_%a.err \
        validation.slurm
done

