# 人群疏散路径优化研究

基于社会力模型(SFM)和PPO强化学习的人群疏散仿真与优化系统。

## 项目简介

本项目以成都东客站地铁出站口为研究场景，实现了：

1. **社会力模型 (SFM)**: 模拟行人运动行为，包括驱动力、社会排斥力、障碍物排斥力
2. **PPO强化学习**: 动态优化出口引导策略，减少拥堵，提高疏散效率
3. **Pygame可视化**: 实时展示疏散过程，支持有/无PPO引导对比

### 场景参数

| 参数 | 值 |
|------|-----|
| 场景尺寸 | 60m × 40m |
| 出口数量 | 3个 (A、B、C) |
| 闸机通道 | 5个 |
| 柱子 | 6个 |
| 行人数量 | 80人 |

## 环境要求

- Python 3.10 - 3.11 (不支持 3.12+)
- macOS / Linux / Windows

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd crowd-evacuation
```

### 2. 创建虚拟环境

**使用 venv:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

**或使用 conda:**
```bash
conda create -n crowd-evacuation python=3.11
conda activate crowd-evacuation
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
pip install pygame  # 可视化需要
```

### 4. 验证安装

```bash
python test_env.py
```

## 使用指南

### 运行地铁站疏散仿真（无PPO）

```bash
python examples/pygame_metro_station.py
```

**操作说明:**
- `空格键`: 暂停/继续
- `R键`: 重新开始仿真
- `ESC`: 退出

### 训练PPO模型

**训练地铁站专用模型（推荐）:**
```bash
python examples/train_ppo_metro.py
```

训练完成后，模型保存至 `outputs/models/ppo_metro.zip`

**训练参数:**
- 训练步数: 50,000
- 观测空间: 8维 (3出口密度 + 3出口拥堵度 + 剩余比例 + 时间比例)
- 动作空间: Discrete(3) - 选择推荐出口 A/B/C

### 运行PPO智能引导仿真

```bash
python examples/pygame_metro_with_ppo.py
```

**操作说明:**
- `空格键`: 暂停/继续
- `P键`: 开关PPO智能引导
- `R键`: 重新开始仿真
- `ESC`: 退出

**颜色说明:**
- 蓝色: 正常行走
- 橙色: 速度较慢
- 红色: 拥堵状态
- 紫色圆圈: PPO推荐的出口

## 项目结构

```
crowd-evacuation/
├── src/                          # 源代码
│   ├── sfm/
│   │   └── social_force.py       # 社会力模型实现
│   ├── simulation/
│   │   ├── evacuation_env.py     # 基础RL环境 (2出口)
│   │   └── metro_evacuation_env.py # 地铁站RL环境 (3出口)
│   ├── ml/
│   │   ├── data_processor.py     # 数据处理
│   │   └── gbm_predictor.py      # GBM预测器
│   └── utils/
│       └── config.py             # 配置工具
│
├── examples/                     # 示例脚本
│   ├── pygame_metro_station.py   # 地铁站可视化 (无PPO)
│   ├── pygame_metro_with_ppo.py  # 地铁站可视化 (PPO引导)
│   ├── train_ppo.py              # 基础PPO训练
│   ├── train_ppo_metro.py        # 地铁站PPO训练
│   └── demo_sfm.py               # SFM演示
│
├── outputs/                      # 输出目录
│   ├── models/                   # 训练模型
│   │   ├── ppo_metro.zip         # 地铁站PPO模型 (3出口)
│   │   └── ppo_evacuation.zip    # 基础PPO模型 (2出口)
│   └── figures/                  # 生成图表
│
├── configs/                      # 配置文件
├── data/                         # 数据文件
├── tests/                        # 测试代码
├── requirements.txt              # 依赖列表
└── pyproject.toml                # 项目配置
```

## 核心模块说明

### 社会力模型 (SFM)

位置: `src/sfm/social_force.py`

实现 Helbing 社会力模型，计算行人受力:
- **驱动力**: 驱使行人向目标移动
- **社会力**: 行人之间的排斥力
- **障碍物力**: 墙壁、柱子等的排斥力

```python
from sfm.social_force import SocialForceModel, Pedestrian

sfm = SocialForceModel(tau=0.5, A=2000.0, B=0.08)
sfm.add_pedestrian(Pedestrian(id=0, position=[10, 10], velocity=[0, 0], target=[50, 20]))
sfm.add_obstacle(start=[0, 0], end=[60, 0])  # 添加墙壁
sfm.step(dt=0.1)  # 仿真一步
```

### 强化学习环境

位置: `src/simulation/metro_evacuation_env.py`

基于 Gymnasium 接口的地铁站疏散环境:

```python
from simulation.metro_evacuation_env import MetroEvacuationEnv

env = MetroEvacuationEnv(n_pedestrians=80, max_steps=800)
obs, info = env.reset()
action = env.action_space.sample()  # 0=出口A, 1=出口B, 2=出口C
obs, reward, terminated, truncated, info = env.step(action)
```

**奖励函数:**
- 疏散奖励: +10 × 新疏散人数
- 拥堵惩罚: -2 × 总拥堵度
- 时间惩罚: -0.1/步
- 完成奖励: +100
- 均衡奖励: 鼓励各出口分流

## 常见问题

### Q: pygame 无法显示中文?

确保系统安装了中文字体。macOS 默认有 PingFang SC，Linux 可安装:
```bash
sudo apt install fonts-noto-cjk  # Ubuntu/Debian
```

### Q: 训练速度慢?

可以调整 `train_ppo_metro.py` 中的参数:
- 减少 `n_pedestrians` (如 40)
- 减少 `total_timesteps` (如 20000)

### Q: 如何对比有无PPO的效果?

运行 `pygame_metro_with_ppo.py`，按 `P` 键可实时切换PPO开关，观察疏散效率差异。

## 技术栈

- **仿真核心**: NumPy, SciPy
- **强化学习**: Stable-Baselines3, Gymnasium
- **可视化**: Pygame, Matplotlib
- **机器学习**: Scikit-learn, XGBoost, PyTorch

## 参考文献

1. Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.

## License

MIT License
