# Training Pipeline Guide

本文档介绍如何运行完整的训练流水线。

---

## 快速开始

```bash
# 快速测试（约10-15分钟）
python scripts/run_full_pipeline.py --quick --skip-ablation

# 完整运行（约6-8小时）
python scripts/run_full_pipeline.py
```

---

## Pipeline 步骤

| Step | 名称 | 描述 | 预计时间 |
|------|------|------|----------|
| 1 | density_predictor | 用Jülich真实数据训练ConvLSTM密度预测模型 | 正常15min / 快速3min |
| 2 | framework_comparison | 研究框架对比实验（5组方案） | 正常2-3h / 快速10min |
| 3 | ablation | 消融实验（17组对比，可跳过） | 正常4h / 快速30min |
| 4 | ppo_training | PPO训练（CPU耗时长） | 正常2h / 快速20min |
| 5 | generalization_test | 泛化性验证（小/中/大流量） | 约10min |

---

## 命令行参数

```bash
python scripts/run_full_pipeline.py [OPTIONS]

Options:
  --quick           快速模式（减少训练步数和实验数量）
  --skip-ablation   跳过消融实验（Step 3）
  --start-step N    从第N步开始
  --only-step N     只执行第N步
```

---

## 运行示例

### 1. 完整流水线（正式实验）

```bash
python scripts/run_full_pipeline.py
```

预计耗时：6-8小时

### 2. 快速验证（测试代码）

```bash
python scripts/run_full_pipeline.py --quick --skip-ablation
```

预计耗时：30-45分钟

### 3. 只训练密度预测模型

```bash
python scripts/run_full_pipeline.py --only-step 1
```

### 4. 只运行对比实验

```bash
# 快速模式：baseline vs full，2 episodes
python scripts/run_full_pipeline.py --only-step 2 --quick

# 完整模式：5方案 × 3流量 × 5 episodes = 75次实验
python scripts/run_full_pipeline.py --only-step 2
```

### 5. 从PPO训练开始（跳过已完成的步骤）

```bash
python scripts/run_full_pipeline.py --start-step 4
```

### 6. 跳过耗时的消融实验

```bash
python scripts/run_full_pipeline.py --skip-ablation
```

---

## 输出文件

运行完成后，输出文件结构：

```
outputs/
├── pipeline_results/
│   └── pipeline_log_YYYYMMDD_HHMMSS.json   # 执行日志
├── models/
│   ├── density_predictor.pt                 # 密度预测模型
│   ├── density_predictor.json               # 训练历史
│   └── ppo_large_station_small.zip          # PPO模型
├── framework_comparison/
│   ├── comparison_report.json               # 对比实验汇总
│   └── detailed_results.json                # 详细结果
└── ablation/
    └── summary_report.json                  # 消融实验汇总
```

---

## 对比实验说明

Step 2 运行5组方案对比（参考 `docs/new_station_plan.md` 6.1节）：

| 方案 | 描述 |
|------|------|
| baseline_sfm | 原始SFM，随机动作（基线） |
| sfm_prediction | SFM + 密度预测 |
| sfm_routing | SFM + 动态分流 |
| sfm_full | SFM + 预测 + 分流（完整方案） |
| sfm_ppo | SFM + PPO引导 |

### 单独运行对比实验

```bash
# 快速模式
python examples/run_framework_comparison.py --quick

# 指定方案和流量
python examples/run_framework_comparison.py \
    --methods baseline_sfm sfm_full sfm_ppo \
    --flow-levels small medium \
    --episodes 3

# 完整实验
python examples/run_framework_comparison.py --episodes 10
```

---

## 可视化

训练完成后，可以使用可视化脚本查看效果：

```bash
# 基线方案
python scripts/visualize_pedestrian_flow.py --method baseline --flow medium

# 完整方案（SFM+预测+分流）
python scripts/visualize_pedestrian_flow.py --method full --flow large

# PPO引导（需要先训练模型）
python scripts/visualize_pedestrian_flow.py --method ppo --flow large
```

**可视化快捷键**：
- `1-5`: 切换方案
- `Space`: 暂停/继续
- `R`: 重置
- `Esc`: 退出

---

## 验证清单

运行完成后检查：

- [ ] `outputs/models/density_predictor.pt` 存在
- [ ] `outputs/models/ppo_large_station_small.zip` 存在
- [ ] `outputs/framework_comparison/comparison_report.json` 存在
- [ ] 泛化性测试输出：
  - 小流量疏散率 ≥ 95%
  - 中流量疏散率 ≥ 90%
  - 大流量疏散率 ≥ 85%

---

## 常见问题

### Q: 运行太慢怎么办？

使用快速模式：
```bash
python scripts/run_full_pipeline.py --quick --skip-ablation
```

### Q: 某一步失败了怎么办？

查看日志文件 `outputs/pipeline_results/pipeline_log_*.json`，然后从失败的步骤重新开始：
```bash
python scripts/run_full_pipeline.py --start-step N
```

### Q: 如何只重跑对比实验？

```bash
python scripts/run_full_pipeline.py --only-step 2
```

### Q: GPU加速？

- 密度预测训练会自动使用GPU（如果可用）
- PPO训练在CPU上运行更快（MLP策略）
- SFM仿真支持GPU加速（自动检测）

---

## 日志格式

`pipeline_log_*.json` 示例：

```json
{
  "pipeline_start": "2025-01-24T14:30:00",
  "pipeline_end": "2025-01-24T17:45:00",
  "mode": "quick",
  "skip_ablation": true,
  "total_duration_seconds": 2700,
  "steps": [
    {
      "step_id": 1,
      "name": "density_predictor",
      "success": true,
      "duration_seconds": 180,
      "output_file": "outputs/models/density_predictor.pt",
      "output_exists": true
    },
    ...
  ]
}
```
