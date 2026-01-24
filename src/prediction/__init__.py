"""
密度场预测模块

通过ConvLSTM预测未来密度分布，为动态分流提供决策支持。

核心组件：
- DensityPredictor: 密度预测模型
- ConvLSTM: 时空序列预测网络
- DensityDataCollector: 训练数据收集器

设计原则：
- 网格结构固定（30×16），不随人数变化
- 密度归一化到[0,1]（基于最大安全密度4.0人/m²）
- 输入/输出与具体人数解耦，保证泛化性
"""

from .density_predictor import DensityFieldPredictor
from .conv_lstm import ConvLSTMCell, ConvLSTM, DensityPredictorNet, DensityPredictorLite
from .data_collector import DensityDataCollector

__all__ = [
    'DensityFieldPredictor',
    'ConvLSTMCell',
    'ConvLSTM',
    'DensityPredictorNet',
    'DensityPredictorLite',
    'DensityDataCollector',
]

# 常量定义
GRID_SIZE = (30, 16)          # 固定网格 (150m/5m × 80m/5m)
CELL_SIZE = 5.0               # 每格5m×5m
MAX_SAFE_DENSITY = 4.0        # 最大安全密度 (人/m²)
MAX_VELOCITY = 2.0            # 最大速度 (m/s)
SCENE_SIZE = (150.0, 80.0)    # 场景尺寸 (m)
