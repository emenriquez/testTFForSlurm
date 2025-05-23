#!/bin/bash

# SLURM Batch Script for Fine-Tuning DistilBERT on Emotion Dataset (PyTorch)
# This script submits a job to SLURM to run torch_finetune_emotion.py on a GPU node.
# It also measures and prints the total training time.
#
# Usage:
#   sbatch slurm_large_ex.sh

# Set the job's name
#SBATCH --job-name=finetune-emotion

# Set the job's output file and path (%j = job ID)
#SBATCH --output=finetune-emotion.out.%j

# Set the job's error output file and path (%j = job ID)
#SBATCH --error=finetune-emotion.err.%j

# Request number of nodes (1 node)
#SBATCH -N 1

# Request partition (queue) to run on (e.g., GPU queue)
#SBATCH -p gpua30q

# Request number and type of GPUs (1 GPU)
#SBATCH --gres=gpu:1

# Set a time limit for the job (hh:mm:ss)
#SBATCH --time=2:00:00

# Load the CUDA module
module load cuda/12.3

# Print which GPUs are visible to this job
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate the appropriate Python environment (edit as needed)
ENV_PATH="/shared/cs4341/bin/activate"
echo "Activating Pytorch environment"
source $ENV_PATH

# Record start time
start_time=$(date +%s)

# Run the PyTorch fine-tuning script
echo "Running torch_finetune_emotion.py"
python3 ~/testTFForSlurm/torch_finetune_emotion.py

# Record end time
end_time=$(date +%s)

# Calculate and print elapsed time
elapsed=$((end_time - start_time))
echo "Total training time: $elapsed seconds ($((elapsed/60)) min $((elapsed%60)) sec)"

echo "Deactivating environment"
deactivate

echo "Done."
