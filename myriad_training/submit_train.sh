#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=12:00:00
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 8
#$ -l mem=4G
#$ -N ds_mask_lora
#$ -cwd
#$ -o train_output_$JOB_ID.log
#$ -e train_error_$JOB_ID.log

module unload python3/recommended
module load python/3.11.4
module load cuda/11.8

source /home/ucjvhai/Scratch/llm_env/bin/activate

export BNB_CUDA_VERSION=118
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/shared/ucl/apps/cuda/11.8.0/gnu-10.2.0/lib64

export HF_HOME=/home/ucjvhai/Scratch/huggingface_cache

llamafactory-cli train deepseek_mask_lora.yaml