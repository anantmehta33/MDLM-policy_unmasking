#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --open-mode=append
#SBATCH --nodes=4
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=sudoku-policy
#SBATCH --output="output/train_sudoku_%j.out"
#SBATCH --ntasks=8
#SBATCH --mem=120G

set -euo pipefail
module purge
module load WebProxy
module load CUDA/12.4.0
module load GCC/11.3.0
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

# Activate your env before launch if your cluster requires it.
# Non-interactive shells (SLURM) do not load conda init hooks automatically.
source /scratch/user/ajayjagan2511/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/user/ajayjagan2511/miniconda3/envs/d1

# In SLURM, BASH_SOURCE[0] may point to a temporary copy under /var/spool,
# so do not derive repo paths from script location.
REPO_ROOT="${REPO_ROOT:-/scratch/user/ajayjagan2511/d1}"
WORKDIR="${REPO_ROOT}/eval/policy_sudoku"
ACCEL_CONFIG="${REPO_ROOT}/diffu-grpo/accelerate.yaml"

TRAIN_CSV="${REPO_ROOT}/dataset/4x4_train_sudoku.csv"
MODEL_NAME="GSAI-ML/LLaDA-1.5"

# Configurable reverse steps for both training and policy-guided inference.
REVERSE_STEPS=16
EPOCHS=20
BATCH_SIZE=8
LR=3e-4
NUM_WORKERS=8
SEED=42

RUN_NAME="sudoku_policy_rs${REVERSE_STEPS}_bs${BATCH_SIZE}"
SAVE_DIR="${WORKDIR}/checkpoints/${RUN_NAME}"
SAMPLE_PUZZLE="3102200002100320"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
# Avoid port collision across concurrent jobs.
MAIN_PORT="${MAIN_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export LOGDIR="${WORKDIR}/slurm/${SLURM_JOB_ID}"
mkdir -p "${LOGDIR}"
mkdir -p "${SAVE_DIR}"

echo "LOGDIR=${LOGDIR}"
echo "WORKDIR=${WORKDIR}"
echo "ACCEL_CONFIG=${ACCEL_CONFIG}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "PYTHON=$(which python)"

cd "${WORKDIR}"

# Command 1: Policy training with accelerate.
if [ "${NUM_GPUS}" -gt 1 ]; then
  srun --ntasks=1 --nodes=1 --output "${LOGDIR}/train_%j.out" \
    python -m accelerate.commands.launch \
      --config_file "${ACCEL_CONFIG}" \
      --multi_gpu \
      --num_processes "${NUM_GPUS}" \
      --main_process_port "${MAIN_PORT}" \
      train_policy_sudoku.py \
        --model_name "${MODEL_NAME}" \
        --train_csv "${TRAIN_CSV}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --reverse_steps "${REVERSE_STEPS}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --device cuda \
        --save_dir "${SAVE_DIR}"
else
  srun --ntasks=1 --nodes=1 --output "${LOGDIR}/train_%j.out" \
    python -m accelerate.commands.launch \
      --config_file "${ACCEL_CONFIG}" \
      --num_processes 1 \
      --main_process_port "${MAIN_PORT}" \
      train_policy_sudoku.py \
        --model_name "${MODEL_NAME}" \
        --train_csv "${TRAIN_CSV}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --reverse_steps "${REVERSE_STEPS}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --device cuda \
        --save_dir "${SAVE_DIR}"
fi

# Command 2: Policy-guided sampled inference using the trained checkpoint.
# This runs single-process intentionally to avoid duplicate sampling logs from all ranks.
srun --ntasks=1 --nodes=1 --output "${LOGDIR}/infer_%j.out" \
  python -m accelerate.commands.launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes 1 \
    --main_process_port "$((MAIN_PORT + 1))" \
    policy_guided_remasking.py \
      --checkpoint "${SAVE_DIR}/policy_best.pt" \
      --puzzle "${SAMPLE_PUZZLE}" \
      --reverse_steps "${REVERSE_STEPS}" \
      --device cuda

echo "Completed training + inference. Check logs under ${LOGDIR}"
