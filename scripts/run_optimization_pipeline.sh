#!/bin/bash
# 优化实验完整执行脚本
# 用法: nohup bash scripts/run_optimization_pipeline.sh > optimization.log 2>&1 &

set -e  # 遇到错误停止

echo "=========================================="
echo "优化实验开始: $(date)"
echo "=========================================="

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "项目目录: $PROJECT_ROOT"

# Step 1: B组统一奖励评估
echo ""
echo "=========================================="
echo "[Step 1/3] B组统一奖励评估"
echo "开始时间: $(date)"
echo "=========================================="
python examples/evaluate_with_unified_reward.py --episodes 10
echo "[Step 1/3] 完成: $(date)"

# Step 2: 多规模验证 (small + medium)
echo ""
echo "=========================================="
echo "[Step 2/3] 多规模验证 (small + medium)"
echo "开始时间: $(date)"
echo "=========================================="
python examples/run_multi_scale_validation.py \
    --groups V1 V2 \
    --scales small medium \
    --quick
echo "[Step 2/3] 完成: $(date)"

# Step 3: 多规模验证 (large) - 可选，耗时较长
echo ""
echo "=========================================="
echo "[Step 3/3] 多规模验证 (large scale)"
echo "开始时间: $(date)"
echo "=========================================="
python examples/run_multi_scale_validation.py \
    --groups V1 V2 \
    --scales large
echo "[Step 3/3] 完成: $(date)"

echo ""
echo "=========================================="
echo "所有实验完成: $(date)"
echo "=========================================="
echo "输出文件:"
echo "  - outputs/ablation/unified_reward_evaluation.json"
echo "  - outputs/multi_scale/<timestamp>/"
