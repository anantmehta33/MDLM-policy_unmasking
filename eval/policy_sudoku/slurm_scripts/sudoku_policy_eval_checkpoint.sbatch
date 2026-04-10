#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sudoku-policy-eval
#SBATCH --output="output/eval_sudoku_%j.out"
#SBATCH --error="error/eval_sudoku_%j.out"
#SBATCH --ntasks=1
#SBATCH --mem=120G

set -euo pipefail
module purge
module load WebProxy
module load CUDA/12.4.0
module load GCC/11.3.0
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

source /scratch/user/ajayjagan2511/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/user/ajayjagan2511/miniconda3/envs/d1

REPO_ROOT="${REPO_ROOT:-/scratch/user/ajayjagan2511/d1}"
WORKDIR="${REPO_ROOT}/eval/policy_sudoku"
ACCEL_CONFIG="${REPO_ROOT}/diffu-grpo/accelerate.yaml"
TEST_CSV="${TEST_CSV:-${REPO_ROOT}/dataset/4x4_test_sudoku.csv}"

# Pass checkpoint path either as env var CHECKPOINT_PATH or as first script argument.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "ERROR: Provide checkpoint path via CHECKPOINT_PATH env var or first positional argument."
  echo "Example: sbatch sudoku_policy_eval_checkpoint.sbatch /path/to/policy_best.pt"
  exit 2
fi

REVERSE_STEPS="${REVERSE_STEPS:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"

MAIN_PORT="${MAIN_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
LOGDIR="${WORKDIR}/slurm/${SLURM_JOB_ID}"
RESULT_JSON="${LOGDIR}/eval_results_${SLURM_JOB_ID}.json"

mkdir -p "${LOGDIR}"

echo "LOGDIR=${LOGDIR}"
echo "WORKDIR=${WORKDIR}"
echo "ACCEL_CONFIG=${ACCEL_CONFIG}"
echo "PYTHON=$(which python)"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "TEST_CSV=${TEST_CSV}"

cd "${WORKDIR}"

srun --ntasks=1 --nodes=1 --output "${LOGDIR}/eval_${SLURM_JOB_ID}.out" \
  python -m accelerate.commands.launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes 1 \
    --main_process_port "${MAIN_PORT}" \
    eval_checkpoint_sudoku.py \
      --checkpoint "${CHECKPOINT_PATH}" \
      --test_csv "${TEST_CSV}" \
      --batch_size "${BATCH_SIZE}" \
      --reverse_steps "${REVERSE_STEPS}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --device cuda \
      --save_json "${RESULT_JSON}"

echo "Evaluation complete."
echo "Detailed log: ${LOGDIR}/eval_${SLURM_JOB_ID}.out"
echo "Result JSON: ${RESULT_JSON}"
