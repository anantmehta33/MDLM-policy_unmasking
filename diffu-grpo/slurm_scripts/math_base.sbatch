#!/bin/bash
#SBATCH --job-name=try
#SBATCH --time=1:00:00
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=50
#SBATCH --mem=120G       
#SBATCH --output=/work/nvme/bdgk/anant/d1/%x_%j.log  
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --account=bcjx-delta-gpu


export LOGDIR=checkpoints/slurm/${SLURM_JOB_ID}
mkdir -p $LOGDIR
echo $LOGDIR


DATASET="math"
RUN_NAME=${DATASET}_base_bs12
MODEL_PATH=GSAI-ML/LLaDA-1.5 
NUM_ITER=12

srun --output ${LOGDIR}/grpo_%j.out \
    accelerate launch \
        --config_file ./accelerate_a100.yaml \
        --main_process_port 12346 ../diffu_grpo_train.py \
        --config ./train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir checkpoints/$RUN_NAME