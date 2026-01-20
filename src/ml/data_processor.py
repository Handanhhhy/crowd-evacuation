"""
轨迹数据处理模块
将原始轨迹数据转换为机器学习训练特征
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional


class TrajectoryDataProcessor:
    """轨迹数据处理器"""

    def __init__(self, obs_len: int = 8, pred_len: int = 12):
        """
        Args:
            obs_len: 观测序列长度（用于预测的历史帧数）
            pred_len: 预测序列长度（需要预测的未来帧数）
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载轨迹数据文件

        格式: frame_id, ped_id, x, y (制表符分隔)
        """
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['frame', 'ped_id', 'x', 'y']
        )
        return df

    def extract_sequences(self, df: pd.DataFrame) -> List[np.ndarray]:
        """提取连续轨迹序列

        Returns:
            sequences: List of arrays, each with shape (seq_len, 2)
        """
        sequences = []

        # 按行人ID分组
        for ped_id, group in df.groupby('ped_id'):
            group = group.sort_values('frame')
            coords = group[['x', 'y']].values

            # 滑动窗口提取序列
            if len(coords) >= self.seq_len:
                for i in range(len(coords) - self.seq_len + 1):
                    seq = coords[i:i + self.seq_len]
                    sequences.append(seq)

        return sequences

    def compute_features(self, trajectory: np.ndarray) -> dict:
        """计算单条轨迹的特征

        Args:
            trajectory: shape (seq_len, 2) 的轨迹坐标

        Returns:
            特征字典
        """
        obs = trajectory[:self.obs_len]  # 观测部分
        pred = trajectory[self.obs_len:]  # 预测部分（真值）

        # 计算速度
        velocities = np.diff(obs, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)

        # 计算加速度
        accelerations = np.diff(velocities, axis=0)

        # 计算方向变化
        directions = np.arctan2(velocities[:, 1], velocities[:, 0])
        direction_changes = np.diff(directions)

        features = {
            # 位置特征
            'pos_x': obs[-1, 0],
            'pos_y': obs[-1, 1],

            # 速度特征
            'vel_x': velocities[-1, 0],
            'vel_y': velocities[-1, 1],
            'speed_mean': np.mean(speeds),
            'speed_std': np.std(speeds),
            'speed_last': speeds[-1] if len(speeds) > 0 else 0,

            # 加速度特征
            'acc_x': accelerations[-1, 0] if len(accelerations) > 0 else 0,
            'acc_y': accelerations[-1, 1] if len(accelerations) > 0 else 0,

            # 方向特征
            'direction': directions[-1] if len(directions) > 0 else 0,
            'direction_change_mean': np.mean(np.abs(direction_changes)) if len(direction_changes) > 0 else 0,

            # 轨迹形状特征
            'displacement': np.linalg.norm(obs[-1] - obs[0]),
            'path_length': np.sum(speeds),
        }

        # 目标：预测下一步位置
        targets = {
            'target_x': pred[0, 0],  # 下一帧x
            'target_y': pred[0, 1],  # 下一帧y
            'target_vx': pred[0, 0] - obs[-1, 0],  # 下一步速度x
            'target_vy': pred[0, 1] - obs[-1, 1],  # 下一步速度y
        }

        return features, targets

    def prepare_dataset(
        self,
        file_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """准备训练数据集

        Returns:
            X: 特征矩阵
            y: 目标矩阵
            feature_names: 特征名列表
            target_names: 目标名列表
        """
        # 加载数据
        df = self.load_data(file_path)
        print(f"加载数据: {len(df)} 条记录")

        # 提取序列
        sequences = self.extract_sequences(df)
        print(f"提取序列: {len(sequences)} 条轨迹")

        # 计算特征
        all_features = []
        all_targets = []

        for seq in sequences:
            features, targets = self.compute_features(seq)
            all_features.append(features)
            all_targets.append(targets)

        # 转换为DataFrame
        features_df = pd.DataFrame(all_features)
        targets_df = pd.DataFrame(all_targets)

        X = features_df.values
        y = targets_df.values
        feature_names = list(features_df.columns)
        target_names = list(targets_df.columns)

        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")

        return X, y, feature_names, target_names


def compute_neighbor_features(
    current_pos: np.ndarray,
    neighbor_positions: np.ndarray,
    max_neighbors: int = 5
) -> np.ndarray:
    """计算邻居特征（用于考虑周围人群密度）

    Args:
        current_pos: 当前行人位置 (2,)
        neighbor_positions: 邻居位置 (N, 2)
        max_neighbors: 最大考虑邻居数

    Returns:
        neighbor_features: 邻居特征向量
    """
    if len(neighbor_positions) == 0:
        return np.zeros(max_neighbors * 2 + 2)

    # 计算到所有邻居的距离和方向
    diffs = neighbor_positions - current_pos
    distances = np.linalg.norm(diffs, axis=1)

    # 按距离排序，取最近的邻居
    sorted_idx = np.argsort(distances)[:max_neighbors]

    features = []
    for i in range(max_neighbors):
        if i < len(sorted_idx):
            idx = sorted_idx[i]
            features.extend([diffs[idx, 0], diffs[idx, 1]])
        else:
            features.extend([0.0, 0.0])

    # 添加密度特征
    density = len(neighbor_positions)
    avg_distance = np.mean(distances) if len(distances) > 0 else 0

    features.extend([density, avg_distance])

    return np.array(features)
