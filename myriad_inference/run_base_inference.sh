#!/bin/bash -l

#$ -l gpu=1
#$ -l mem=32G
#$ -l h_rt=01:00:00

#$ -N ds_base_scene
#$ -j y
#$ -wd /myriadfs/home/ucjvhai/Scratch/LLaMA-Factory

module load python/3.11.4
module load cuda/11.8
source /myriadfs/home/ucjvhai/Scratch/llm_env/bin/activate

echo "Starting Base Model Inference (With Scenario) Task at $(date)"

python batch_inference.py \
    --base_model_path /myriadfs/home/ucjvhai/Scratch/open_source_llms/deepseek-moe-16b-chat-masked \
    --input_file /acfs/users/ucjvhai/test_set_with_scenario.xlsx \
    --output_file test_result_base_final.xlsx

echo "All evaluations finished successfully at $(date)"