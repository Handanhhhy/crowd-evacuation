"""机器学习模块

包含以下组件:
- TrajectoryDataProcessor: 轨迹数据处理
- GBMPredictor: 梯度提升模型行为预测
- SocialLSTM: Social-LSTM轨迹预测 (Alahi et al. 2016)
- TrajectronPlusPlus: Trajectron++多模态轨迹预测 (Salzmann et al. 2020)
- TrajectoryPredictor: 统一轨迹预测接口
- TrajectoryDataset: 统一数据加载器
"""
from .data_processor import TrajectoryDataProcessor

# GBM requires joblib - optional
try:
    from .gbm_predictor import GBMPredictor
except ImportError:
    GBMPredictor = None

# Trajectory prediction modules
from .trajectory_predictor import SocialLSTM, TrajectoryPredictor
from .trajectron import TrajectronPlusPlus, TrajectronLoss
from .data_loader import TrajectoryDataset, MultiDatasetLoader
