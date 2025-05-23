#!/bin/bash

# SLURM Sample Job Script
# This script demonstrates how to submit a simple job to a SLURM-managed HPC cluster.
# It requests 1 GPU node, activates a TensorFlow environment, and runs a Python script.
#
# Usage:
#   sbatch slurm_sample.sh

### Set the job's name (appears in queue and output files)
#SBATCH --job-name=myFirstJob

### Set the job's output file and path (%j = job ID)
#SBATCH --output=myFirstJob.out.%j

### Set the job's error output file and path (%j = job ID)
#SBATCH --error=myFirstJob.err.%j

### Request number of nodes (here, 1 node)
#SBATCH -N 1

### Request partition (queue) to run on (e.g., GPU queue)
#SBATCH -p gpua30q

### Request number and type of GPUs (here, 1 GPU)
#SBATCH --gres=gpu:1

### Set a time limit for the job (hh:mm:ss)
#SBATCH --time=1:00:00

# Print which GPUs are visible to this job
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Define environment and script paths for clarity
TF_ENV_PATH="/shared/tensorflow-2.6.2/tf_env/bin/activate"
PYTHON_SCRIPT="~/testTFForSlurm/testTF.py"

# Activate the TensorFlow environment
echo "Activating TensorFlow-2.6.2 environment"
source $TF_ENV_PATH

# Run the Python script
echo "Running testTF.py"
python3 $PYTHON_SCRIPT

# Deactivate the environment
echo "Deactivating TensorFlow-2.6.2 environment"
deactivate

echo "Done."
