#!/bin/bash
set -euo pipefail  # 严格模式：未定义变量报错、管道失败报错、错误立即退出

# ===================== 核心配置（按需修改）=====================
# 1. 定义8个DNN模型名称
DNN_MODELS=(
    "llama2_7B"
    "llama2_13B"
    "llama2_34B"
    "bert"
    "vgg16"
    "resnet50"
    "alexnet"
    "retinanet"
    "unet"
)

# 2. 定义需要自动运行的加速器列表（Simba + Gemmini）
ACCELERATORS=(
    # "Simba"
    "Gemmini"
)

# 3. 输出根目录（所有加速器+模型的日志/时间记录）
OUTPUT_ROOT="./mapping_constraint/dnn_batch_results"

# 4. 总时间统计文件（记录所有加速器+模型的运行数据）
GLOBAL_TIME_LOG="${OUTPUT_ROOT}/dnn_global_run_summary.log"

# ===================== 初始化工作 =====================
# 创建输出根目录
mkdir -p "${OUTPUT_ROOT}"

# 初始化全局时间统计文件（表头，包含加速器列）
echo -e "加速器\tDNN模型\t开始时间\t结束时间\t总耗时(秒)\t运行状态\t日志文件路径" > "${GLOBAL_TIME_LOG}"

# ===================== 批量执行（加速器×模型）=====================
# 遍历每个加速器
for ACCELERATOR in "${ACCELERATORS[@]}"; do
    echo -e "\n========================================================"
    echo "🚀 开始运行加速器：${ACCELERATOR}"
    echo "========================================================"

    # 遍历当前加速器下的所有DNN模型
    for DNN in "${DNN_MODELS[@]}"; do
        echo -e "\n--------------------------------------------------------"
        echo "🔄 运行 DNN 模型：${DNN} | 加速器：${ACCELERATOR}"
        echo "--------------------------------------------------------"

        # 1. 创建当前加速器+模型的独立输出目录
        MODEL_OUTPUT_DIR="${OUTPUT_ROOT}/${ACCELERATOR}/${DNN}"
        mkdir -p "${MODEL_OUTPUT_DIR}"

        # 2. 定义日志文件路径（区分加速器和模型）
        MODEL_LOG="${MODEL_OUTPUT_DIR}/${DNN}_${ACCELERATOR}_run_hf.log"

        # 3. 记录开始时间
        START_TIMESTAMP=$(date +%s)
        START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
        STATUS="成功"

        # 4. 执行 ./run_saturn.sh（传入模型名+加速器）
        if ./run_mapping_constraint.sh "${DNN}" "${ACCELERATOR}" 1 > "${MODEL_LOG}" 2>&1; then
            echo "✅ ${DNN} (${ACCELERATOR}) 运行完成！"
        else
            STATUS="失败"
            echo "❌ ${DNN} (${ACCELERATOR}) 运行失败！错误日志：${MODEL_LOG}"
        fi

        # 5. 记录结束时间和耗时
        END_TIMESTAMP=$(date +%s)
        END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
        ELAPSED_TIME=$((END_TIMESTAMP - START_TIMESTAMP))

        # 6. 写入全局时间统计文件
        echo -e "${ACCELERATOR}\t${DNN}\t${START_TIME}\t${END_TIME}\t${ELAPSED_TIME}\t${STATUS}\t${MODEL_LOG}" >> "${GLOBAL_TIME_LOG}"

        echo -e "📊 ${DNN} (${ACCELERATOR}) 运行结束 | 耗时：${ELAPSED_TIME} 秒 | 状态：${STATUS}"
        echo "--------------------------------------------------------"
    done

    echo -e "========================================================"
    echo "✅ 加速器 ${ACCELERATOR} 所有模型运行完成！"
    echo "========================================================"
done

# ===================== 运行完成汇总 =====================
echo -e "\n🎉 所有加速器+DNN模型组合运行完成！"
echo "📋 全局时间统计文件：${GLOBAL_TIME_LOG}"
echo "📁 所有日志根目录：${OUTPUT_ROOT}"

# 格式化打印汇总表（更易读）
echo -e "\n【全局运行汇总表】"
column -t -s $'\t' "${GLOBAL_TIME_LOG}"

# 可选：输出每个加速器的总耗时统计
echo -e "\n【各加速器总耗时统计】"
for ACC in "${ACCELERATORS[@]}"; do
    # 提取当前加速器的所有耗时，求和
    TOTAL_TIME=$(grep "^${ACC}\t" "${GLOBAL_TIME_LOG}" | awk -F '\t' '{sum+=$5} END {print sum}')
    echo "🔧 加速器 ${ACC} 总耗时：${TOTAL_TIME} 秒（约 $((TOTAL_TIME/60)) 分钟）"
done