"""
训练数据收集器

从SFM仿真中收集密度场训练数据。

功能：
1. 收集仿真帧数据（密度、流场、出口距离）
2. 构建序列数据集
3. 支持多episode数据收集
4. 数据增强（可选）

参考文档: docs/new_station_plan.md 密度场预测模块 TODO
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, asdict

from .density_predictor import (
    DensityFieldPredictor,
    DensityField,
    GRID_SIZE,
    CELL_SIZE,
    MAX_SAFE_DENSITY,
    MAX_VELOCITY,
    SCENE_SIZE,
)


@dataclass
class Episode:
    """一个episode的数据"""
    frames: List[DensityField]
    metadata: Dict
    
    def __len__(self):
        return len(self.frames)


class DensityDataCollector:
    """训练数据收集器
    
    从SFM仿真中收集密度场数据，用于训练ConvLSTM。
    
    使用示例:
        collector = DensityDataCollector(exits=exits)
        
        # 仿真循环
        for step in range(max_steps):
            collector.collect_frame(pedestrians, current_time)
            sfm.step(dt)
        
        # 保存episode
        collector.save_episode("episode_001")
        
        # 构建数据集
        dataset = collector.build_dataset(
            seq_length=10,
            pred_horizon=50,
        )
    """
    
    def __init__(
        self,
        exits: List[Dict],
        grid_size: Tuple[int, int] = GRID_SIZE,
        cell_size: float = CELL_SIZE,
        scene_size: Tuple[float, float] = SCENE_SIZE,
        max_safe_density: float = MAX_SAFE_DENSITY,
        max_velocity: float = MAX_VELOCITY,
        save_dir: str = "outputs/training_data",
    ):
        """
        Args:
            exits: 出口列表
            grid_size: 网格尺寸
            cell_size: 单元格大小
            scene_size: 场景尺寸
            max_safe_density: 最大安全密度
            max_velocity: 最大速度
            save_dir: 数据保存目录
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.scene_size = scene_size
        self.max_safe_density = max_safe_density
        self.max_velocity = max_velocity
        self.exits = exits
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 预计算出口距离场
        self.exit_distance_field = self._compute_exit_distance_field()
        
        # 当前episode数据
        self.current_frames: List[DensityField] = []
        self.episode_metadata: Dict = {}
        
        # 已保存的episode
        self.episodes: List[Episode] = []
        
    def _compute_exit_distance_field(self) -> np.ndarray:
        """预计算出口距离场"""
        grid_w, grid_h = self.grid_size
        distance_field = np.full((grid_w, grid_h), np.inf)
        
        for i in range(grid_w):
            for j in range(grid_h):
                x = (i + 0.5) * self.cell_size
                y = (j + 0.5) * self.cell_size
                cell_pos = np.array([x, y])
                
                min_dist = np.inf
                for exit_info in self.exits:
                    exit_pos = exit_info.get('position', exit_info.get('pos', np.zeros(2)))
                    if isinstance(exit_pos, (list, tuple)):
                        exit_pos = np.array(exit_pos)
                    dist = np.linalg.norm(cell_pos - exit_pos)
                    if dist < min_dist:
                        min_dist = dist
                
                distance_field[i, j] = min_dist
        
        valid_distances = distance_field[np.isfinite(distance_field)]
        if len(valid_distances) > 0:
            max_dist = np.max(valid_distances)
            if max_dist > 0:
                distance_field = distance_field / max_dist
        else:
            # 如果没有有效距离（例如没有出口），全设为1.0
            distance_field[:] = 1.0
            
        distance_field = np.clip(distance_field, 0.0, 1.0)
        
        return distance_field
    
    def _pos_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """场景坐标转网格坐标"""
        i = int(x / self.cell_size)
        j = int(y / self.cell_size)
        i = max(0, min(i, self.grid_size[0] - 1))
        j = max(0, min(j, self.grid_size[1] - 1))
        return i, j
    
    def collect_frame(
        self,
        pedestrians: List,
        timestamp: float,
    ) -> DensityField:
        """收集一帧数据
        
        Args:
            pedestrians: 行人列表（可以是Pedestrian对象或dict）
            timestamp: 时间戳
            
        Returns:
            DensityField: 收集的密度场
        """
        grid_w, grid_h = self.grid_size
        
        # 初始化网格
        density_grid = np.zeros((grid_w, grid_h))
        flow_x_grid = np.zeros((grid_w, grid_h))
        flow_y_grid = np.zeros((grid_w, grid_h))
        count_grid = np.zeros((grid_w, grid_h))
        
        # 统计每个网格的行人
        for ped in pedestrians:
            # 支持对象和字典两种格式
            if hasattr(ped, 'position'):
                pos = ped.position
                vel = ped.velocity
            else:
                pos = ped.get('position', ped.get('pos', np.zeros(2)))
                vel = ped.get('velocity', ped.get('vel', np.zeros(2)))
            
            if isinstance(pos, (list, tuple)):
                pos = np.array(pos)
            if isinstance(vel, (list, tuple)):
                vel = np.array(vel)
                
            i, j = self._pos_to_grid(pos[0], pos[1])
            
            count_grid[i, j] += 1
            flow_x_grid[i, j] += vel[0]
            flow_y_grid[i, j] += vel[1]
        
        # 计算密度和平均流场
        cell_area = self.cell_size ** 2
        
        for i in range(grid_w):
            for j in range(grid_h):
                count = count_grid[i, j]
                if count > 0:
                    local_density = count / cell_area
                    density_grid[i, j] = min(local_density / self.max_safe_density, 1.0)
                    flow_x_grid[i, j] = (flow_x_grid[i, j] / count) / self.max_velocity
                    flow_y_grid[i, j] = (flow_y_grid[i, j] / count) / self.max_velocity
        
        # 流场归一化
        flow_x_grid = np.clip(flow_x_grid, -1.0, 1.0) * 0.5 + 0.5
        flow_y_grid = np.clip(flow_y_grid, -1.0, 1.0) * 0.5 + 0.5
        
        field = DensityField(
            density=density_grid,
            flow_x=flow_x_grid,
            flow_y=flow_y_grid,
            exit_distance=self.exit_distance_field.copy(),
            timestamp=timestamp,
        )
        
        self.current_frames.append(field)
        return field

    def collect_frame_gpu(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        timestamp: float,
    ) -> DensityField:
        """收集一帧数据（GPU加速版）
        
        直接利用PyTorch Tensor计算密度场，避免CPU-GPU传输和Python循环。
        
        Args:
            positions: 位置张量 [N, 2]
            velocities: 速度张量 [N, 2]
            timestamp: 时间戳
        """
        grid_w, grid_h = self.grid_size
        device = positions.device
        
        # 计算网格索引
        # x_idx = floor(x / cell_size)
        grid_indices = (positions / self.cell_size).long()
        
        # 过滤超出边界的点
        mask = (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < grid_w) & \
               (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < grid_h)
        
        valid_indices = grid_indices[mask]
        valid_velocities = velocities[mask]
        
        # 初始化网格张量
        density_grid = torch.zeros((grid_w, grid_h), device=device)
        flow_x_grid = torch.zeros((grid_w, grid_h), device=device)
        flow_y_grid = torch.zeros((grid_w, grid_h), device=device)
        count_grid = torch.zeros((grid_w, grid_h), device=device)
        
        if len(valid_indices) > 0:
            # 使用 scatter_add 或类似操作聚合
            # 这里为了简单，先把二维索引转为一维线性索引
            linear_indices = valid_indices[:, 0] * grid_h + valid_indices[:, 1]
            
            # 统计人数
            count_grid.view(-1).scatter_add_(0, linear_indices, torch.ones_like(linear_indices, dtype=torch.float))
            
            # 累加速度
            flow_x_grid.view(-1).scatter_add_(0, linear_indices, valid_velocities[:, 0])
            flow_y_grid.view(-1).scatter_add_(0, linear_indices, valid_velocities[:, 1])
            
            # 计算平均值和密度
            cell_area = self.cell_size ** 2
            has_ped = count_grid > 0
            
            # 密度 = 人数 / 面积 / 最大安全密度
            density_grid[has_ped] = (count_grid[has_ped] / cell_area) / self.max_safe_density
            density_grid = torch.clamp(density_grid, 0.0, 1.0)
            
            # 平均速度
            flow_x_grid[has_ped] /= count_grid[has_ped]
            flow_y_grid[has_ped] /= count_grid[has_ped]
            
            # 归一化速度
            flow_x_grid = flow_x_grid / self.max_velocity
            flow_y_grid = flow_y_grid / self.max_velocity
            
            flow_x_grid = torch.clamp(flow_x_grid, -1.0, 1.0) * 0.5 + 0.5
            flow_y_grid = torch.clamp(flow_y_grid, -1.0, 1.0) * 0.5 + 0.5
            
        # 转回CPU numpy
        field = DensityField(
            density=density_grid.cpu().numpy(),
            flow_x=flow_x_grid.cpu().numpy(),
            flow_y=flow_y_grid.cpu().numpy(),
            exit_distance=self.exit_distance_field.copy(),
            timestamp=timestamp,
        )
        
        self.current_frames.append(field)
        return field
    
    def start_episode(self, metadata: Optional[Dict] = None):
        """开始新的episode"""
        self.current_frames = []
        self.episode_metadata = metadata or {}
    
    def end_episode(self) -> Episode:
        """结束当前episode"""
        episode = Episode(
            frames=self.current_frames.copy(),
            metadata=self.episode_metadata.copy(),
        )
        self.episodes.append(episode)
        
        # 清空当前数据
        self.current_frames = []
        self.episode_metadata = {}
        
        return episode
    
    def save_episode(self, name: str):
        """保存当前episode到文件"""
        if not self.current_frames:
            print(f"[DataCollector] 警告: 当前episode为空，跳过保存")
            return
        
        episode_dir = self.save_dir / name
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存帧数据
        frames_data = []
        for frame in self.current_frames:
            frames_data.append({
                'density': frame.density.tolist(),
                'flow_x': frame.flow_x.tolist(),
                'flow_y': frame.flow_y.tolist(),
                'exit_distance': frame.exit_distance.tolist(),
                'timestamp': frame.timestamp,
            })
        
        with open(episode_dir / "frames.pkl", 'wb') as f:
            pickle.dump(frames_data, f)
        
        # 保存元数据
        with open(episode_dir / "metadata.json", 'w') as f:
            json.dump(self.episode_metadata, f, indent=2)
        
        print(f"[DataCollector] 保存episode: {name} ({len(self.current_frames)}帧)")
    
    def load_episode(self, name: str) -> Episode:
        """从文件加载episode"""
        episode_dir = self.save_dir / name
        
        with open(episode_dir / "frames.pkl", 'rb') as f:
            frames_data = pickle.load(f)
        
        with open(episode_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        frames = []
        for fd in frames_data:
            frame = DensityField(
                density=np.array(fd['density']),
                flow_x=np.array(fd['flow_x']),
                flow_y=np.array(fd['flow_y']),
                exit_distance=np.array(fd['exit_distance']),
                timestamp=fd['timestamp'],
            )
            frames.append(frame)
        
        episode = Episode(frames=frames, metadata=metadata)
        self.episodes.append(episode)
        
        print(f"[DataCollector] 加载episode: {name} ({len(frames)}帧)")
        return episode
    
    def load_all_episodes(self) -> List[Episode]:
        """加载目录下所有episode"""
        self.episodes = []
        
        for episode_dir in sorted(self.save_dir.iterdir()):
            if episode_dir.is_dir() and (episode_dir / "frames.pkl").exists():
                try:
                    self.load_episode(episode_dir.name)
                except Exception as e:
                    print(f"[DataCollector] 加载失败 {episode_dir.name}: {e}")
        
        print(f"[DataCollector] 共加载 {len(self.episodes)} 个episode")
        return self.episodes
    
    def build_dataset(
        self,
        seq_length: int = 10,
        pred_horizon: int = 50,
        stride: int = 5,
        train_ratio: float = 0.8,
    ) -> Tuple['DensitySequenceDataset', 'DensitySequenceDataset']:
        """构建训练和验证数据集
        
        Args:
            seq_length: 输入序列长度（历史帧数）
            pred_horizon: 预测步长（未来帧数）
            stride: 采样步长
            train_ratio: 训练集比例
            
        Returns:
            (train_dataset, val_dataset)
        """
        if not self.episodes:
            raise ValueError("没有可用的episode数据")
        
        # 收集所有序列
        all_sequences = []
        
        for episode in self.episodes:
            frames = episode.frames
            n_frames = len(frames)
            
            # 需要 seq_length + pred_horizon 帧才能构建一个样本
            min_length = seq_length + pred_horizon
            if n_frames < min_length:
                continue
            
            # 滑动窗口采样
            for start_idx in range(0, n_frames - min_length + 1, stride):
                input_frames = frames[start_idx:start_idx + seq_length]
                target_frame = frames[start_idx + seq_length + pred_horizon - 1]
                
                all_sequences.append({
                    'input': input_frames,
                    'target': target_frame,
                })
        
        if not all_sequences:
            raise ValueError("无法构建有效序列，请检查episode长度")
        
        # 打乱并分割
        np.random.shuffle(all_sequences)
        
        split_idx = int(len(all_sequences) * train_ratio)
        train_sequences = all_sequences[:split_idx]
        val_sequences = all_sequences[split_idx:]
        
        train_dataset = DensitySequenceDataset(train_sequences)
        val_dataset = DensitySequenceDataset(val_sequences)
        
        print(f"[DataCollector] 数据集构建完成:")
        print(f"  - 训练集: {len(train_dataset)} 样本")
        print(f"  - 验证集: {len(val_dataset)} 样本")
        
        return train_dataset, val_dataset
    
    def get_statistics(self) -> Dict:
        """获取数据统计"""
        total_frames = sum(len(ep) for ep in self.episodes)
        
        if self.episodes:
            avg_density_values = []
            max_density_values = []
            
            for ep in self.episodes:
                for frame in ep.frames:
                    avg_density_values.append(np.mean(frame.density))
                    max_density_values.append(np.max(frame.density))
        
            return {
                'n_episodes': len(self.episodes),
                'total_frames': total_frames,
                'avg_frames_per_episode': total_frames / len(self.episodes),
                'avg_density': float(np.mean(avg_density_values)),
                'max_density': float(np.max(max_density_values)),
                'current_episode_frames': len(self.current_frames),
            }
        
        return {
            'n_episodes': 0,
            'total_frames': 0,
            'current_episode_frames': len(self.current_frames),
        }


class DensitySequenceDataset(Dataset):
    """密度序列数据集
    
    用于PyTorch DataLoader的数据集类。
    """
    
    def __init__(self, sequences: List[Dict]):
        """
        Args:
            sequences: 序列列表 [{'input': [DensityField], 'target': DensityField}, ...]
        """
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_tensor: [seq_len, 4, 30, 16]
            target_tensor: [1, 30, 16]
        """
        seq = self.sequences[idx]
        
        # 输入序列
        input_frames = []
        for frame in seq['input']:
            data = np.stack([
                frame.density,
                frame.flow_x,
                frame.flow_y,
                frame.exit_distance,
            ], axis=0)
            input_frames.append(data)
        
        input_tensor = torch.from_numpy(np.stack(input_frames, axis=0)).float()
        
        # 目标（预测密度）
        target_tensor = torch.from_numpy(seq['target'].density[np.newaxis, :, :]).float()
        
        return input_tensor, target_tensor


def create_dataloader(
    dataset: DensitySequenceDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
