#!/bin/bash
set -euo pipefail

# ===================== 配置项 =====================
# 输出根目录（存储两个脚本的运行日志和汇总）
OUTPUT_ROOT="./all_scripts_results"
# 总统计文件（记录每个脚本的运行时间）
SCRIPT_TIME_LOG="${OUTPUT_ROOT}/scripts_run_summary.log"

# ===================== 初始化 =====================
mkdir -p "${OUTPUT_ROOT}"
echo -e "脚本名称\t开始时间\t结束时间\t总耗时(秒)\t运行状态" > "${SCRIPT_TIME_LOG}"

# ===================== 执行 run_hf_all.sh =====================
echo -e "\n========================================================"
echo "📌 开始执行：run_hf_all.sh"
echo "========================================================"

# 记录开始时间
START_HF=$(date +%s)
START_HF_STR=$(date +"%Y-%m-%d %H:%M:%S")
STATUS_HF="成功"

# 执行脚本，输出重定向到独立日志
HF_LOG="${OUTPUT_ROOT}/run_hf_all.log"
if ./run_hf_all.sh > "${HF_LOG}" 2>&1; then
    echo "✅ run_hf_all.sh 执行完成！日志：${HF_LOG}"
else
    STATUS_HF="失败"
    echo "❌ run_hf_all.sh 执行失败！日志：${HF_LOG}"
fi

# 记录结束时间和耗时
END_HF=$(date +%s)
END_HF_STR=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_HF=$((END_HF - START_HF))
echo -e "run_hf_all.sh\t${START_HF_STR}\t${END_HF_STR}\t${ELAPSED_HF}\t${STATUS_HF}" >> "${SCRIPT_TIME_LOG}"

# ===================== 执行 run_saturn_all.sh =====================
echo -e "\n========================================================"
echo "📌 开始执行：run_saturn_all.sh"
echo "========================================================"

# 记录开始时间
START_SATURN=$(date +%s)
START_SATURN_STR=$(date +"%Y-%m-%d %H:%M:%S")
STATUS_SATURN="成功"

# 执行脚本，输出重定向到独立日志
SATURN_LOG="${OUTPUT_ROOT}/run_saturn_all.log"
if ./run_saturn_all.sh > "${SATURN_LOG}" 2>&1; then
    echo "✅ run_saturn_all.sh 执行完成！日志：${SATURN_LOG}"
else
    STATUS_SATURN="失败"
    echo "❌ run_saturn_all.sh 执行失败！日志：${SATURN_LOG}"
fi

# 记录结束时间和耗时
END_SATURN=$(date +%s)
END_SATURN_STR=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_SATURN=$((END_SATURN - START_SATURN))
echo -e "run_saturn_all.sh\t${START_SATURN_STR}\t${END_SATURN_STR}\t${ELAPSED_SATURN}\t${STATUS_SATURN}" >> "${SCRIPT_TIME_LOG}"

# ===================== 汇总 =====================
echo -e "\n========================================================"
echo "🏁 所有脚本执行完成！"
echo "========================================================"
echo "📋 脚本运行统计：${SCRIPT_TIME_LOG}"
echo "📁 脚本日志目录：${OUTPUT_ROOT}"

# 格式化打印统计结果
echo -e "\n【脚本运行汇总表】"
column -t -s $'\t' "${SCRIPT_TIME_LOG}"

# 计算总耗时
TOTAL_ELAPSED=$((ELAPSED_HF + ELAPSED_SATURN))
echo -e "\n⏱️  总耗时：${TOTAL_ELAPSED} 秒（约 $((TOTAL_ELAPSED/60)) 分钟）"