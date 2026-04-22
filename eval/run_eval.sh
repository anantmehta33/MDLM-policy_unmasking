#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=run_eval
#SBATCH --time=1:00:00                     
#SBATCH --ntasks=1                         
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G                             
#SBATCH --output=/work/nvme/bdgk/anant/d1/eval/anant_logs/logs/%x_%j.log  
#SBATCH --gres=gpu:4
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bdgk-dtai-gh

set -euo pipefail

# module purge
# module load WebProxy
# module load CUDA/12.4.0
# module load GCC/11.3.0
# export http_proxy=http://10.73.132.63:8080
# export https_proxy=http://10.73.132.63:8080

# Activate your env before launch if your cluster requires it.
# Non-interactive shells (SLURM) do not load conda init hooks automatically.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate d1

# Hardcoded evaluation config for SLURM runs.
# Update these paths/values directly in this file before submission.
EVAL_MODE="policy"  # options: policy, base_low_confidence

# Policy sweep config used only when EVAL_MODE=policy.
POLICY_CKPTS=(
  "/work/nvme/bdgk/anant/d1/policy_training/checkpoints/sudoku_policy_rs16_bs8/policy_best_GSAI-ML_LLaDA-1.5.pt"
)
DIFFUSION_STEPS=(2 4 8)

# Kept for base_low_confidence mode.
BASE_DIFFUSION_STEPS=8

# Used only in policy mode. Base mode uses eval/sudoku.py default test set (4x4_test_sudoku.csv).
SUDOKU_CSV="/work/nvme/bdgk/anant/d1/dataset/4x4_test_sudoku.csv"
GPU_ID="0"

# if [ "$EVAL_MODE" = "policy" ]; then
#   if [ ! -f "$POLICY_CKPT" ]; then
#     echo "Policy checkpoint not found: $POLICY_CKPT"
#     exit 1
#   fi

#   if [ ! -f "$SUDOKU_CSV" ]; then
#     echo "Sudoku CSV not found: $SUDOKU_CSV"
#     exit 1
#   fi
# fi

# Configuration variables
GPU_IDS=($GPU_ID)

MASTER_PORT=29411

# Arrays of tasks and generation lengths
#TASKS=("countdown" "sudoku" "math" "gsm8k")
# We only plan to use "soduko" as evaluation task
TASKS=("sudoku")
GEN_LENGTHS=(16)

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
# echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
# echo "Eval mode: $EVAL_MODE"
# echo "Policy checkpoints: ${POLICY_CKPTS[*]}"
# echo "Sudoku CSV: $SUDOKU_CSV"
# echo "PYTHON=$(which python)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"

    if [ "$EVAL_MODE" = "policy" ]; then
      for policy_ckpt in "${POLICY_CKPTS[@]}"; do
        if [ ! -f "$policy_ckpt" ]; then
          echo "Policy checkpoint not found: $policy_ckpt"
          exit 1
        fi

        for diffusion_steps in "${DIFFUSION_STEPS[@]}"; do
          echo "Policy eval: ckpt=$policy_ckpt diffusion_steps=$diffusion_steps"

          CUDA_VISIBLE_DEVICES=$GPU_LIST python -m torch.distributed.run \
            --nproc_per_node $NUM_GPUS \
            --master_port $MASTER_PORT \
            eval.py \
            --dataset $task \
            --batch_size $batch_size \
            --gen_length $gen_length \
            --block_length 16 \
            --diffusion_steps $diffusion_steps \
            --sudoku_csv "$SUDOKU_CSV" \
            --policy_checkpoint_path "$policy_ckpt" \
            --remasking_strategy policy \
            --policy_reward_guided \
            --policy_reward_candidates 4 \
            --output_dir "eval_results" \
            --model_path "GSAI-ML/LLaDA-1.5"
        done
      done
    elif [ "$EVAL_MODE" = "base_low_confidence" ]; then
      CUDA_VISIBLE_DEVICES=$GPU_LIST python -m torch.distributed.run \
        --nproc_per_node $NUM_GPUS \
        --master_port $MASTER_PORT \
        eval.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --sudoku_csv "$SUDOKU_CSV" \
        --block_length 16 \
        --diffusion_steps $BASE_DIFFUSION_STEPS \
        --remasking_strategy low_confidence \
        --output_dir "eval_results" \
        --model_path "GSAI-ML/LLaDA-1.5"
    else
      echo "Unknown EVAL_MODE: $EVAL_MODE"
      exit 1
    fi
  done
done


echo "All evaluations completed!"
