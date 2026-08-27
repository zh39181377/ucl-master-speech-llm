#!/bin/bash -l

# 申请 1 张 GPU，32GB 内存，运行时间限制 2 小时
#$ -l gpu=1
#$ -l mem=32G
#$ -l h_rt=01:00:00

# 设置任务名称
#$ -N ds_lora_eval

# 合并标准输出和错误输出到同一个日志文件
#$ -j y

# 设定工作目录
#$ -wd /myriadfs/home/ucjvhai/Scratch/LLaMA-Factory

# 加载必要的底层模块
module load python/3.11.4
module load cuda/11.8

# 激活虚拟环境
source /myriadfs/home/ucjvhai/Scratch/llm_env/bin/activate

echo "Starting LoRA Inference Task at $(date)"

# 执行带 LoRA 权重的批量推理脚本
python batch_inference.py \
    --base_model_path /myriadfs/home/ucjvhai/Scratch/open_source_llms/deepseek-moe-16b-chat-masked \
    --lora_path /myriadfs/home/ucjvhai/Scratch/outputs/deepseek_mask_lora \
    --input_file /acfs/users/ucjvhai/test_set_with_scenario.xlsx \
    --output_file test_result_lora_final.xlsx

echo "All evaluations finished successfully at $(date)"