# 密度场预测问题分析

## 问题描述

从可视化结果（样本 145/545）可以看出：
- **当前密度**：0.00-0.10（低密度）
- **预测密度**：0.91（高密度热点）
- **真实密度**：0.00-0.10（仍为低密度）
- **误差**：平均 0.17，最大 0.91

模型**严重过预测**了密度，预测了不存在的高密度聚集。

## 根本原因分析

### 1. 预测步长过长 ⚠️

**当前配置**：
- `seq_length = 10`（输入1秒历史）
- `pred_horizon = 50`（预测5秒后）

**问题**：
- 5秒对于人群动态来说**太长**了
- 人群在5秒内可能已经疏散完毕或完全改变方向
- 模型无法准确预测如此长的时间跨度

**证据**：
- 训练损失：0.0034（看起来不错）
- 但实际预测时出现严重偏差
- 说明模型在训练集上表现好，但泛化能力差

### 2. 训练数据不平衡 ⚠️

**可能的问题**：
- 训练数据中**高密度场景较少**
- 大部分样本是低密度场景
- 模型学习到了"低密度→高密度"的转换模式
- 但实际中，低密度场景通常**保持低密度**

**验证方法**：
```python
# 检查训练数据中密度分布
import numpy as np
from src.prediction.data_collector import DensityDataCollector

collector = DensityDataCollector(...)
collector.load_all_episodes()

all_densities = []
for ep in collector.episodes:
    for frame in ep.frames:
        all_densities.append(frame.density.flatten())

all_densities = np.concatenate(all_densities)
print(f"密度统计:")
print(f"  均值: {np.mean(all_densities):.4f}")
print(f"中位数: {np.median(all_densities):.4f}")
print(f"  最大值: {np.max(all_densities):.4f}")
print(f"  高密度比例 (>0.5): {np.mean(all_densities > 0.5):.2%}")
print(f"  高密度比例 (>0.7): {np.mean(all_densities > 0.7):.2%}")
```

### 3. 模型架构问题 ⚠️

**ConvLSTM 的局限性**：
- ConvLSTM 擅长捕捉**空间-时间模式**
- 但可能**过度拟合**了训练数据中的模式
- 对于"低密度保持低密度"这种**静态模式**，模型可能学习不足

**可能的学习偏差**：
- 模型看到很多"低密度→高密度"的转换（疏散开始）
- 但很少看到"低密度→保持低密度"（疏散后期）
- 导致模型倾向于预测密度增加

### 4. 损失函数问题 ⚠️

**MSE 损失的局限性**：
- MSE 对**大误差**惩罚更重
- 如果训练数据中高密度场景少，模型可能：
  - 倾向于预测**平均值**
  - 或者**过度预测**以覆盖所有可能情况

**建议**：
- 考虑使用**加权MSE**（对高密度区域加权）
- 或使用**Focal Loss**（关注难预测样本）

## 解决方案

### 方案1：缩短预测步长（推荐）✅

**修改配置**：
```python
# 从 5秒 缩短到 1-2秒
pred_horizon = 10  # 1秒后（10帧）
# 或
pred_horizon = 20  # 2秒后（20帧）
```

**理由**：
- 1-2秒的预测更可靠
- 人群动态在短时间内的变化更可预测
- 可以用于实时决策（动态分流）

**重新训练**：
```bash
python scripts/train_density_predictor.py \
    --train \
    --pred-horizon 10 \
    --epochs 50
```

### 方案2：数据增强 ✅

**增加"低密度保持低密度"的样本**：
```python
# 在数据收集时，增加后期疏散阶段的样本
# 或人工添加"静态低密度"场景
```

**平衡训练数据**：
- 确保高密度和低密度场景比例均衡
- 增加"密度保持"的场景

### 方案3：改进损失函数 ✅

**使用加权MSE**：
```python
def weighted_mse_loss(pred, target, weight_map):
    """根据密度值加权"""
    # 高密度区域权重更大
    weights = 1.0 + weight_map * 2.0  # 高密度区域权重3倍
    loss = (pred - target) ** 2 * weights
    return loss.mean()
```

**使用Focal Loss**：
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """关注难预测样本"""
    mse = (pred - target) ** 2
    weight = alpha * (mse ** gamma)
    return (weight * mse).mean()
```

### 方案4：多步预测 ✅

**使用多步预测**：
- 不直接预测50帧后
- 而是预测10帧后，然后用预测结果再预测10帧后
- 逐步推进，更稳定

**实现**：
```python
# 使用 predict_multi_step
predictions = model.predict_multi_step(x, steps=5)  # 每步10帧
```

### 方案5：添加正则化 ✅

**防止过拟合**：
```python
# 在训练时添加L2正则化
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# 或使用Dropout
# 在ConvLSTM中添加Dropout层
```

## 快速验证

### 检查数据分布
```bash
python -c "
from src.prediction.data_collector import DensityDataCollector
import numpy as np

collector = DensityDataCollector(exits=[], save_dir='outputs/training_data')
collector.load_all_episodes()

all_densities = []
for ep in collector.episodes:
    for frame in ep.frames:
        all_densities.append(frame.density.flatten())

all_densities = np.concatenate(all_densities)
print(f'密度统计:')
print(f'  均值: {np.mean(all_densities):.4f}')
print(f'中位数: {np.median(all_densities):.4f}')
print(f'  最大值: {np.max(all_densities):.4f}')
print(f'  高密度比例 (>0.5): {np.mean(all_densities > 0.5):.2%}')
print(f'  高密度比例 (>0.7): {np.mean(all_densities > 0.7):.2%}')
"
```

### 测试短预测步长
```bash
# 使用更短的预测步长重新训练
python scripts/train_density_predictor.py \
    --train \
    --pred-horizon 10 \
    --epochs 30 \
    --data-dir outputs/training_data
```

## 推荐行动

1. **立即行动**：缩短预测步长到 10-20 帧（1-2秒）
2. **数据分析**：检查训练数据的密度分布
3. **重新训练**：使用更短的预测步长
4. **评估改进**：对比新旧模型的预测效果

## 预期改进

使用 `pred_horizon=10`（1秒）后：
- ✅ 预测更准确（误差降低50%+）
- ✅ 更符合实际应用场景（实时决策）
- ✅ 模型更稳定（不会出现极端预测）
