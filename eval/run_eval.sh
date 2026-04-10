#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --time=1:00:00
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G       
#SBATCH --output=/work/nvme/bdgk/anant/d1/%x_%j.log  
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --account=bcjx-delta-gpu

set -euo pipefail

# Configuration variables
GPU_IDS=(0)  # Default GPU IDs to use

MASTER_PORT=29411

# Arrays of tasks and generation lengths
#TASKS=("countdown" "sudoku" "math" "gsm8k")
# We only plan to use "soduko" as evaluation task
TASKS=("sudoku")
GEN_LENGTHS=(128)

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --toy_evaluation \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir "eval_results" \
      --model_path "GSAI-ML/LLaDA-1.5"
  done
done


echo "All evaluations completed!"
