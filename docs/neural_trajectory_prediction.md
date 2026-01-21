# 预测性疏通系统 - 神经网络轨迹预测

## 概述

本模块实现了基于 Social-LSTM 的神经网络轨迹预测系统，用于预测行人未来轨迹并主动避免角落拥堵。

**文献参考**: Alahi et al. 2016 "Social LSTM: Human Trajectory Prediction in Crowded Spaces"

## 系统架构

```
┌────────────────────────────────────────────────────────────┐
│                    预测性疏通系统                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Social-LSTM  │───▶│ 轨迹预测     │───▶│ 拥堵预测     │ │
│  │ (神经网络)   │    │ (未来4.8秒)  │    │ (哪个出口)   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ 角落检测     │    │ 碰撞预测     │    │ 智能重分配   │ │
│  │ (预测是否    │    │ (会不会卡住) │    │ (改道决策)   │ │
│  │  走向死角)   │    │              │    │              │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 文件结构

```
crowd-evacuation/
├── src/ml/
│   ├── trajectory_predictor.py    # Social-LSTM 模型和预测器
│   ├── gbm_predictor.py           # GBM 行为预测 (已有)
│   └── data_processor.py          # 数据处理 (已有)
├── examples/
│   ├── train_trajectory.py        # 轨迹预测模型训练脚本
│   └── pygame_metro_with_ppo.py   # 可视化仿真 (已更新)
├── outputs/models/
│   └── social_lstm.pt             # 训练好的模型
└── docs/
    └── neural_trajectory_prediction.md  # 本文档
```

## 快速开始

### 1. 训练轨迹预测模型

```bash
cd crowd-evacuation
source .venv/bin/activate

# 训练 Social-LSTM 模型
python examples/train_trajectory.py
```

训练完成后，模型保存到 `outputs/models/social_lstm.pt`

### 2. 运行仿真

```bash
# 运行带神经网络预测的仿真
python examples/pygame_metro_with_ppo.py
```

### 3. 控制键

| 按键 | 功能 |
|------|------|
| SPACE | 暂停/继续 |
| P | 切换 PPO 引导 |
| T | 切换颜色模式 (按类型/按速度) |
| V | 切换预测轨迹显示 |
| R | 重新开始 |
| ESC | 退出 |

## 核心组件

### 1. SocialLSTM 模型

```python
from ml.trajectory_predictor import SocialLSTM

model = SocialLSTM(
    obs_len=8,           # 观测8帧 (3.2秒)
    pred_len=12,         # 预测12帧 (4.8秒)
    embedding_dim=64,    # 位置嵌入维度
    hidden_dim=128,      # LSTM隐藏层维度
    pool_dim=64,         # Social Pooling输出维度
    grid_size=4,         # 社会池化网格大小
    neighborhood_size=2.0  # 邻域范围 (米)
)
```

**架构**:
1. **位置嵌入**: (x, y) → embedding_dim
2. **LSTM编码器**: 处理观测轨迹
3. **Social Pooling**: 捕捉行人间交互
4. **LSTM解码器**: 生成预测轨迹
5. **输出层**: hidden_dim → (x, y)

### 2. TrajectoryPredictor 预测器

```python
from ml.trajectory_predictor import TrajectoryPredictor

predictor = TrajectoryPredictor(
    model_path='outputs/models/social_lstm.pt',  # 模型路径
    obs_len=8,
    pred_len=12,
    device='cpu'  # 或 'cuda'
)

# 更新历史轨迹
predictor.update_history(ped_id, position)

# 批量预测
predictions = predictor.predict_all_trajectories(pedestrians)

# 角落检测
is_trapped, corner = predictor.detect_corner_trap(trajectory, corners)
```

### 3. 集成到 MetroEvacuationEnv

```python
from simulation.metro_evacuation_env import MetroEvacuationEnv

env = MetroEvacuationEnv(
    n_pedestrians=80,
    enable_neural_prediction=True,
    trajectory_model_path='outputs/models/social_lstm.pt'
)

# step() 自动执行:
# 1. 轨迹预测
# 2. 角落检测
# 3. 智能重分配
obs, reward, done, truncated, info = env.step(action)

# info 包含:
# - corner_avoided_this_step: 本步避免角落的行人数
# - neural_prediction: 是否使用神经网络预测
```

## 训练配置

```python
config = {
    'obs_len': 8,           # 观测8帧 (3.2秒 @ 2.5Hz)
    'pred_len': 12,         # 预测12帧 (4.8秒)
    'embedding_dim': 64,
    'hidden_dim': 128,
    'pool_dim': 64,
    'grid_size': 4,
    'neighborhood_size': 2.0,
    'dropout': 0.0,
    'learning_rate': 1e-3,
    'batch_size': 8,
    'epochs': 50,
}
```

## 评估指标

- **ADE (Average Displacement Error)**: 平均位移误差
- **FDE (Final Displacement Error)**: 最终位移误差

```python
from ml.trajectory_predictor import compute_ade_fde

ade, fde = compute_ade_fde(predicted_trajectory, ground_truth)
```

## 与现有系统的对比

| 特性 | 当前（规则计算） | 改进后（神经网络） |
|-----|-----------------|-------------------|
| 预测方法 | 线性外推 `pos + vel * t` | Social-LSTM |
| 考虑周围人 | ❌ | ✅ Social Pooling |
| 考虑障碍物 | ❌ | ✅ 从数据学习 |
| 预测准确性 | 低 | 高 (ADE < 0.5m) |
| 角落检测 | 无 | ✅ 基于轨迹预测 |
| 计算成本 | 低 | 中 (GPU加速) |

## 角落陷阱位置

系统预定义的角落陷阱位置:

```python
corner_traps = [
    np.array([60, 40]),   # Exit C 上方右角落 (主要问题区域)
    np.array([60, 0]),    # 右下角落
    np.array([55, 16]),   # 楼梯旁边
    np.array([55, 24]),   # 楼梯旁边
]
```

## 故障排除

### 神经网络模型加载失败

如果遇到模型加载问题，系统会自动回退到线性外推:

```
轨迹预测器: 使用线性外推 (未找到神经网络模型)
```

解决方法:
1. 确保运行了训练脚本: `python examples/train_trajectory.py`
2. 检查模型文件是否存在: `outputs/models/social_lstm.pt`

### 内存不足

对于大量行人的场景，可以:
1. 减少 `n_pedestrians`
2. 使用 `enable_neural_prediction=False` 回退到线性外推

## API 参考

### SocialLSTM

```python
class SocialLSTM(nn.Module):
    def forward(self, obs_traj, seq_start_end) -> torch.Tensor:
        """
        Args:
            obs_traj: (obs_len, total_peds, 2) 观测轨迹
            seq_start_end: 每个场景的行人索引范围
        Returns:
            pred_traj: (pred_len, total_peds, 2) 预测轨迹
        """
```

### TrajectoryPredictor

```python
class TrajectoryPredictor:
    def update_history(self, ped_id: int, position: np.ndarray) -> None
    def remove_pedestrian(self, ped_id: int) -> None
    def predict_trajectory_linear(self, ped_id, pos, vel, dt=0.4) -> np.ndarray
    def predict_all_trajectories(self, pedestrians, scene_bounds=None) -> Dict[int, np.ndarray]
    def detect_corner_trap(self, trajectory, corners, trap_radius=3.0) -> Tuple[bool, Optional[np.ndarray]]
```

## 参考文献

1. Alahi, A., et al. (2016). "Social LSTM: Human Trajectory Prediction in Crowded Spaces." CVPR.
2. Helbing, D., & Molnar, P. (1995). "Social force model for pedestrian dynamics."
3. ETH/UCY Dataset: 真实行人轨迹数据集
