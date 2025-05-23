# SLURM HPC Tutorial: TensorFlow & PyTorch Examples

This project provides simple examples for running machine learning jobs on a SLURM-managed high-performance computing (HPC) cluster. It includes both a small TensorFlow test and a larger PyTorch fine-tuning job, along with ready-to-use SLURM batch scripts.

## Contents

- `testTF.py`: Minimal TensorFlow script to test GPU access.
- `torch_finetune_emotion.py`: PyTorch script for fine-tuning DistilBERT on the Hugging Face 'emotion' dataset.
- `slurm_small_example.sh`: SLURM script to run `testTF.py` (TensorFlow, quick test).
- `slurm_large_ex.sh` or `slurm_torch_finetune_emotion.sh`: SLURM script to run `torch_finetune_emotion.py` (PyTorch, larger job).

## Usage

1. **Edit the SLURM scripts if needed** (e.g., environment paths, partition names).
2. **Submit a job to the cluster:**
   ```bash
   sbatch slurm_small_example.sh      # For the small TensorFlow test
   sbatch slurm_large_ex.sh          # For the PyTorch fine-tuning job
   ```
3. **Check output files** (e.g., `myFirstJob.out.<jobid>`, `finetune-emotion.out.<jobid>`) for logs and results.

## What Each Script Does

- **slurm_small_example.sh**
  - Requests 1 GPU node
  - Activates a TensorFlow environment
  - Runs `testTF.py` to verify GPU access

- **torch_finetune_emotion.py**
  - Loads the 'emotion' dataset
  - Tokenizes and fine-tunes DistilBERT for emotion classification
  - Evaluates and saves the trained model
  - Prints a sample prediction

- **slurm_large_ex.sh** (or similar)
  - Requests 1 GPU node
  - Activates the environment
  - Runs `torch_finetune_emotion.py`
  - Measures and prints total training time

## Getting Help
If you have questions or run into issues, contact your instructor or cluster administrator.
