#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=04:00:00          # 预留 4 小时
#$ -l gpu=1                  # 申请 1 张 GPU
#$ -ac allow=LUV             # 限定 GPU 节点类型
#$ -pe smp 1                 # 单核心
#$ -l mem=32G                # 内存 32G
#$ -N ds_evaluate_models     # 任务名称
#$ -wd /myriadfs/home/ucjvhai/Scratch/LLaMA-Factory  # 工作目录
#$ -j y                      # 合并日志

echo "Starting evaluation job on $(hostname)..."

# ==========================================
# 【关键修复】先加载集群系统提供的 Python 与 CUDA 模块
# ==========================================
module unload python3/recommended
module load python/3.11.4
module load cuda/11.8

# 1. 激活 Python 虚拟环境
source /home/ucjvhai/Scratch/llm_env/bin/activate

# 2. 定义路径变量
BASE_MODEL="/myriadfs/home/ucjvhai/Scratch/open_source_llms/deepseek-moe-16b-chat-masked"
LORA_WEIGHTS="/myriadfs/home/ucjvhai/Scratch/outputs/deepseek_mask_lora"

# 3. 测试集 Excel 文件路径
TEST_SCENARIO="/acfs/users/ucjvhai/test_set_with_scenario.xlsx"
TEST_NO_SCENARIO="/acfs/users/ucjvhai/test_set_without_scenario.xlsx"

# 任务 1：使用 Base Model 跑带有 Scenario 的测试集
echo "==================================="
echo "Task 1: Base Model + With Scenario"
python batch_inference.py \
    --base_model_path $BASE_MODEL \
    --input_file $TEST_SCENARIO \
    --output_file "Base_Result_with_scenario.xlsx"

# 任务 2：使用 Base Model 跑无 Scenario 的测试集
echo "==================================="
echo "Task 2: Base Model + Without Scenario"
python batch_inference.py \
    --base_model_path $BASE_MODEL \
    --input_file $TEST_NO_SCENARIO \
    --output_file "Base_Result_without_scenario.xlsx"

# 任务 3：使用 LoRA Model 跑带有 Scenario 的测试集
echo "==================================="
echo "Task 3: LoRA Model + With Scenario"
python batch_inference.py \
    --base_model_path $BASE_MODEL \
    --lora_path $LORA_WEIGHTS \
    --input_file $TEST_SCENARIO \
    --output_file "LoRA_Result_with_scenario.xlsx"

# 任务 4：使用 LoRA Model 跑无 Scenario 的测试集
echo "==================================="
echo "Task 4: LoRA Model + Without Scenario"
python batch_inference.py \
    --base_model_path $BASE_MODEL \
    --lora_path $LORA_WEIGHTS \
    --input_file $TEST_NO_SCENARIO \
    --output_file "LoRA_Result_without_scenario.xlsx"

echo "All evaluations finished successfully at $(date)!"