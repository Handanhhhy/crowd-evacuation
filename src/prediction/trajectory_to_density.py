"""
真实轨迹数据到密度场转换器

将 Jülich 等真实轨迹数据转换为密度场训练数据。

支持的数据格式：
- Jülich bottleneck: ped_id frame x y z (空格分隔, cm单位)
- ETH/UCY: frame ped_id x y (制表符分隔, m单位)

功能：
1. 读取轨迹数据文件
2. 自动检测场景范围
3. 计算速度（通过位置差分）
4. 转换为密度场网格
5. 构建序列数据集
6. 保存为与 SFM 数据兼容的格式

参考文档: docs/new_station_plan.md 密度场预测模块 TODO
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import pickle
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .density_predictor import (
    DensityField,
    GRID_SIZE,
    CELL_SIZE,
    MAX_SAFE_DENSITY,
    MAX_VELOCITY,
    SCENE_SIZE,
)
from .data_collector import Episode, DensitySequenceDataset


@dataclass
class TrajectoryConfig:
    """轨迹数据配置"""
    name: str                          # 数据集名称
    format: str                        # 格式: 'juelich', 'eth_ucy'
    unit: str                          # 单位: 'cm', 'm'
    fps: float                         # 帧率
    columns: List[str]                 # 列名
    delimiter: str                     # 分隔符


# 预定义数据集配置
DATASET_CONFIGS = {
    'juelich': TrajectoryConfig(
        name='Jülich Bottleneck',
        format='juelich',
        unit='cm',
        fps=25.0,  # 25 FPS
        columns=['ped_id', 'frame', 'x', 'y', 'z'],
        delimiter=' ',
    ),
    'eth_ucy': TrajectoryConfig(
        name='ETH/UCY',
        format='eth_ucy',
        unit='m',
        fps=2.5,  # 2.5 FPS
        columns=['frame', 'ped_id', 'x', 'y'],
        delimiter='\t',
    ),
}


class TrajectoryToDensityConverter:
    """轨迹到密度场转换器
    
    将真实轨迹数据转换为密度场训练数据。
    
    使用示例:
        converter = TrajectoryToDensityConverter(
            data_format='juelich',
            grid_size=(30, 16),
        )
        
        # 加载并转换数据
        episodes = converter.convert_files(
            ['data/raw/juelich/bottleneck/ao-500-400.txt'],
            save_dir='outputs/training_data/juelich',
        )
    """
    
    def __init__(
        self,
        data_format: str = 'juelich',
        grid_size: Tuple[int, int] = GRID_SIZE,
        cell_size: float = CELL_SIZE,
        max_safe_density: float = MAX_SAFE_DENSITY,
        max_velocity: float = MAX_VELOCITY,
        target_fps: float = 10.0,  # 目标帧率（与SFM仿真对齐: dt=0.1s）
        exit_positions: Optional[List[np.ndarray]] = None,
        auto_detect_exits: bool = True,
    ):
        """
        Args:
            data_format: 数据格式 ('juelich', 'eth_ucy')
            grid_size: 网格尺寸
            cell_size: 单元格大小 (m)
            max_safe_density: 最大安全密度 (人/m²)
            max_velocity: 最大速度 (m/s)
            target_fps: 目标帧率（用于下采样）
            exit_positions: 出口位置列表 (如果为None则自动检测)
            auto_detect_exits: 是否自动检测出口
        """
        self.data_format = data_format
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.max_safe_density = max_safe_density
        self.max_velocity = max_velocity
        self.target_fps = target_fps
        self.exit_positions = exit_positions or []
        self.auto_detect_exits = auto_detect_exits
        
        # 获取数据集配置
        if data_format in DATASET_CONFIGS:
            self.config = DATASET_CONFIGS[data_format]
        else:
            raise ValueError(f"不支持的数据格式: {data_format}，支持: {list(DATASET_CONFIGS.keys())}")
        
        # 场景范围（从数据自动检测）
        self.scene_bounds: Optional[Dict] = None
        self.scene_size: Optional[Tuple[float, float]] = None
        
        # 出口距离场
        self.exit_distance_field: Optional[np.ndarray] = None
        
    def load_trajectory_file(
        self,
        file_path: Union[str, Path],
    ) -> pd.DataFrame:
        """加载轨迹文件
        
        Args:
            file_path: 轨迹文件路径
            
        Returns:
            DataFrame: 包含 ped_id, frame, x, y 的数据框
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        print(f"[TrajectoryConverter] 加载: {file_path.name}")
        
        # 读取数据
        try:
            df = pd.read_csv(
                file_path,
                sep=self.config.delimiter,
                header=None,
                names=self.config.columns,
            )
        except Exception as e:
            # 尝试自动检测分隔符
            for delim in [' ', '\t', ',']:
                try:
                    df = pd.read_csv(
                        file_path,
                        sep=delim,
                        header=None,
                        names=self.config.columns[:4],  # 只取前4列
                    )
                    print(f"  使用分隔符: '{repr(delim)}'")
                    break
                except:
                    continue
            else:
                raise ValueError(f"无法解析文件: {file_path}") from e
        
        # 确保基本列存在
        required_cols = ['ped_id', 'frame', 'x', 'y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")
        
        # 单位转换
        if self.config.unit == 'cm':
            df['x'] = df['x'] / 100.0  # cm -> m
            df['y'] = df['y'] / 100.0
        
        print(f"  行人数: {df['ped_id'].nunique()}")
        print(f"  帧范围: {df['frame'].min()} - {df['frame'].max()}")
        print(f"  X范围: {df['x'].min():.2f} - {df['x'].max():.2f} m")
        print(f"  Y范围: {df['y'].min():.2f} - {df['y'].max():.2f} m")
        
        return df
    
    def _detect_scene_bounds(self, df: pd.DataFrame, padding: float = 1.0):
        """检测场景范围
        
        Args:
            df: 轨迹数据
            padding: 边界填充 (m)
        """
        x_min, x_max = df['x'].min() - padding, df['x'].max() + padding
        y_min, y_max = df['y'].min() - padding, df['y'].max() + padding
        
        self.scene_bounds = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
        }
        
        # 计算场景尺寸
        width = x_max - x_min
        height = y_max - y_min
        self.scene_size = (width, height)
        
        # 更新 cell_size 以适应网格
        self.cell_size = max(width / self.grid_size[0], height / self.grid_size[1])
        
        print(f"[TrajectoryConverter] 场景范围:")
        print(f"  X: [{x_min:.2f}, {x_max:.2f}] m ({width:.2f} m)")
        print(f"  Y: [{y_min:.2f}, {y_max:.2f}] m ({height:.2f} m)")
        print(f"  Cell Size: {self.cell_size:.2f} m")
    
    def _detect_exits(self, df: pd.DataFrame):
        """自动检测出口位置（基于轨迹消失点）
        
        Args:
            df: 轨迹数据
        """
        if not self.auto_detect_exits:
            return
        
        # 找到每个行人的最后位置
        last_positions = df.groupby('ped_id').agg({
            'frame': 'max',
            'x': 'last',
            'y': 'last',
        }).reset_index()
        
        # 聚类最后位置来识别出口
        from scipy.cluster.hierarchy import fcluster, linkage
        
        positions = last_positions[['x', 'y']].values
        
        if len(positions) > 1:
            # 层次聚类
            Z = linkage(positions, method='average')
            # 使用距离阈值（场景尺寸的10%）
            threshold = max(self.scene_size) * 0.1 if self.scene_size else 2.0
            clusters = fcluster(Z, threshold, criterion='distance')
            
            # 计算每个聚类的中心
            unique_clusters = np.unique(clusters)
            self.exit_positions = []
            
            for cluster_id in unique_clusters:
                mask = clusters == cluster_id
                count = np.sum(mask)
                
                # 只有足够多行人离开的位置才被认为是出口
                if count >= max(3, len(positions) * 0.05):
                    center = positions[mask].mean(axis=0)
                    self.exit_positions.append(center)
            
            print(f"[TrajectoryConverter] 自动检测到 {len(self.exit_positions)} 个出口")
            for i, pos in enumerate(self.exit_positions):
                print(f"  出口 {i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
        # 如果没有检测到出口，使用场景边界
        if not self.exit_positions:
            if self.scene_bounds:
                # 默认在y轴最小处设置出口
                center_x = (self.scene_bounds['x_min'] + self.scene_bounds['x_max']) / 2
                self.exit_positions = [np.array([center_x, self.scene_bounds['y_min']])]
                print(f"[TrajectoryConverter] 使用默认出口: ({self.exit_positions[0][0]:.2f}, {self.exit_positions[0][1]:.2f})")
    
    def _compute_exit_distance_field(self):
        """计算出口距离场"""
        if not self.exit_positions:
            # 如果没有出口，使用均匀距离场
            self.exit_distance_field = np.ones(self.grid_size) * 0.5
            return
        
        grid_w, grid_h = self.grid_size
        distance_field = np.full((grid_w, grid_h), np.inf)
        
        for i in range(grid_w):
            for j in range(grid_h):
                # 网格中心世界坐标
                x = self.scene_bounds['x_min'] + (i + 0.5) * self.cell_size
                y = self.scene_bounds['y_min'] + (j + 0.5) * self.cell_size
                cell_pos = np.array([x, y])
                
                # 找最近出口
                min_dist = np.inf
                for exit_pos in self.exit_positions:
                    dist = np.linalg.norm(cell_pos - exit_pos)
                    if dist < min_dist:
                        min_dist = dist
                
                distance_field[i, j] = min_dist
        
        # 归一化到 [0, 1]
        max_dist = np.max(distance_field[np.isfinite(distance_field)])
        if max_dist > 0:
            distance_field = distance_field / max_dist
        distance_field = np.clip(distance_field, 0.0, 1.0)
        
        self.exit_distance_field = distance_field
    
    def _pos_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        if self.scene_bounds is None:
            raise ValueError("场景范围未设置")
        
        i = int((x - self.scene_bounds['x_min']) / self.cell_size)
        j = int((y - self.scene_bounds['y_min']) / self.cell_size)
        i = max(0, min(i, self.grid_size[0] - 1))
        j = max(0, min(j, self.grid_size[1] - 1))
        return i, j
    
    def _compute_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算速度（通过位置差分）
        
        Args:
            df: 轨迹数据
            
        Returns:
            DataFrame: 添加了 vx, vy 列的数据框
        """
        # 按行人和帧排序
        df = df.sort_values(['ped_id', 'frame']).copy()
        
        # 计算时间步长
        dt = 1.0 / self.config.fps
        
        # 计算速度
        df['vx'] = df.groupby('ped_id')['x'].diff() / dt
        df['vy'] = df.groupby('ped_id')['y'].diff() / dt
        
        # 填充第一帧的NaN
        df['vx'] = df['vx'].fillna(0.0)
        df['vy'] = df['vy'].fillna(0.0)
        
        # 限制速度范围（处理异常值）
        speed = np.sqrt(df['vx']**2 + df['vy']**2)
        max_speed = self.max_velocity * 2  # 允许一些超调
        mask = speed > max_speed
        if mask.any():
            scale = max_speed / speed[mask]
            df.loc[mask, 'vx'] *= scale
            df.loc[mask, 'vy'] *= scale
        
        return df
    
    def _downsample_frames(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """下采样帧率到目标FPS
        
        Args:
            df: 轨迹数据
            
        Returns:
            下采样后的数据
        """
        source_fps = self.config.fps
        
        if source_fps <= self.target_fps:
            return df
        
        # 计算采样间隔
        sample_interval = int(source_fps / self.target_fps)
        
        # 获取需要保留的帧
        all_frames = sorted(df['frame'].unique())
        sampled_frames = all_frames[::sample_interval]
        
        df_sampled = df[df['frame'].isin(sampled_frames)].copy()
        
        # 重新编号帧
        frame_map = {f: i for i, f in enumerate(sampled_frames)}
        df_sampled['frame'] = df_sampled['frame'].map(frame_map)
        
        print(f"[TrajectoryConverter] 下采样: {source_fps} -> {self.target_fps} FPS")
        print(f"  帧数: {len(all_frames)} -> {len(sampled_frames)}")
        
        return df_sampled
    
    def _frame_to_density_field(
        self,
        frame_df: pd.DataFrame,
        timestamp: float,
    ) -> DensityField:
        """将单帧数据转换为密度场
        
        Args:
            frame_df: 单帧的行人数据
            timestamp: 时间戳
            
        Returns:
            DensityField
        """
        grid_w, grid_h = self.grid_size
        
        # 初始化网格
        density_grid = np.zeros((grid_w, grid_h))
        flow_x_grid = np.zeros((grid_w, grid_h))
        flow_y_grid = np.zeros((grid_w, grid_h))
        count_grid = np.zeros((grid_w, grid_h))
        
        # 统计每个网格的行人
        for _, row in frame_df.iterrows():
            x, y = row['x'], row['y']
            vx = row.get('vx', 0.0)
            vy = row.get('vy', 0.0)
            
            i, j = self._pos_to_grid(x, y)
            
            count_grid[i, j] += 1
            flow_x_grid[i, j] += vx
            flow_y_grid[i, j] += vy
        
        # 计算密度和平均流场
        cell_area = self.cell_size ** 2
        
        for i in range(grid_w):
            for j in range(grid_h):
                count = count_grid[i, j]
                if count > 0:
                    # 密度
                    local_density = count / cell_area
                    density_grid[i, j] = min(local_density / self.max_safe_density, 1.0)
                    # 平均流场
                    flow_x_grid[i, j] = (flow_x_grid[i, j] / count) / self.max_velocity
                    flow_y_grid[i, j] = (flow_y_grid[i, j] / count) / self.max_velocity
        
        # 流场归一化到 [0, 1]
        flow_x_grid = np.clip(flow_x_grid, -1.0, 1.0) * 0.5 + 0.5
        flow_y_grid = np.clip(flow_y_grid, -1.0, 1.0) * 0.5 + 0.5
        
        return DensityField(
            density=density_grid,
            flow_x=flow_x_grid,
            flow_y=flow_y_grid,
            exit_distance=self.exit_distance_field.copy(),
            timestamp=timestamp,
        )
    
    def convert_file(
        self,
        file_path: Union[str, Path],
        episode_name: Optional[str] = None,
    ) -> Episode:
        """转换单个轨迹文件为Episode
        
        Args:
            file_path: 轨迹文件路径
            episode_name: Episode名称（默认使用文件名）
            
        Returns:
            Episode: 包含密度场帧的Episode
        """
        file_path = Path(file_path)
        if episode_name is None:
            episode_name = file_path.stem
        
        # 1. 加载数据
        df = self.load_trajectory_file(file_path)
        
        # 2. 检测场景范围
        self._detect_scene_bounds(df)
        
        # 3. 检测出口
        self._detect_exits(df)
        
        # 4. 计算出口距离场
        self._compute_exit_distance_field()
        
        # 5. 计算速度
        df = self._compute_velocities(df)
        
        # 6. 下采样
        df = self._downsample_frames(df)
        
        # 7. 转换每帧
        frames: List[DensityField] = []
        all_frames = sorted(df['frame'].unique())
        
        dt = 1.0 / self.target_fps
        
        for frame_idx in tqdm(all_frames, desc=f"Converting {episode_name}"):
            frame_df = df[df['frame'] == frame_idx]
            timestamp = frame_idx * dt
            
            field = self._frame_to_density_field(frame_df, timestamp)
            frames.append(field)
        
        # 创建 Episode
        metadata = {
            'source_file': str(file_path),
            'format': self.data_format,
            'n_frames': len(frames),
            'n_pedestrians': df['ped_id'].nunique(),
            'scene_bounds': self.scene_bounds,
            'cell_size': self.cell_size,
            'target_fps': self.target_fps,
            'grid_size': self.grid_size,
        }
        
        episode = Episode(frames=frames, metadata=metadata)
        
        print(f"[TrajectoryConverter] 转换完成: {episode_name}")
        print(f"  帧数: {len(frames)}")
        print(f"  时长: {len(frames) * dt:.1f}s")
        
        return episode
    
    def convert_files(
        self,
        file_paths: List[Union[str, Path]],
        save_dir: str = "outputs/training_data/juelich",
    ) -> List[Episode]:
        """转换多个轨迹文件
        
        Args:
            file_paths: 轨迹文件路径列表
            save_dir: 保存目录
            
        Returns:
            List[Episode]: Episode列表
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        episodes = []
        
        for file_path in file_paths:
            file_path = Path(file_path)
            episode_name = f"juelich_{file_path.stem}"
            
            try:
                episode = self.convert_file(file_path, episode_name)
                episodes.append(episode)
                
                # 保存 Episode
                self._save_episode(episode, episode_name, save_path)
                
            except Exception as e:
                print(f"[TrajectoryConverter] 转换失败 {file_path}: {e}")
                continue
        
        print(f"\n[TrajectoryConverter] 总计转换 {len(episodes)} 个 Episode")
        
        return episodes
    
    def _save_episode(
        self,
        episode: Episode,
        name: str,
        save_dir: Path,
    ):
        """保存 Episode（与 DensityDataCollector 兼容的格式）"""
        episode_dir = save_dir / name
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存帧数据
        frames_data = []
        for frame in episode.frames:
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
            json.dump(episode.metadata, f, indent=2)
        
        print(f"  保存: {episode_dir}")


def convert_juelich_data(
    input_dir: str = "data/raw/juelich/bottleneck",
    output_dir: str = "outputs/training_data/juelich",
    grid_size: Tuple[int, int] = GRID_SIZE,
) -> List[Episode]:
    """便捷函数：转换 Jülich 瓶颈数据
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        grid_size: 网格尺寸
        
    Returns:
        List[Episode]: 转换的 Episode 列表
    """
    input_path = Path(input_dir)
    
    # 查找所有 .txt 文件
    txt_files = sorted(input_path.glob("*.txt"))
    
    if not txt_files:
        print(f"[Error] 未找到轨迹文件: {input_dir}")
        return []
    
    print(f"[TrajectoryConverter] 找到 {len(txt_files)} 个轨迹文件")
    for f in txt_files:
        print(f"  - {f.name}")
    
    # 创建转换器
    converter = TrajectoryToDensityConverter(
        data_format='juelich',
        grid_size=grid_size,
        target_fps=10.0,  # 与 SFM 仿真对齐
    )
    
    # 转换所有文件
    episodes = converter.convert_files(txt_files, output_dir)
    
    return episodes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="轨迹到密度场转换器")
    parser.add_argument("--input-dir", type=str, default="data/raw/juelich/bottleneck",
                        help="输入目录")
    parser.add_argument("--output-dir", type=str, default="outputs/training_data/juelich",
                        help="输出目录")
    parser.add_argument("--format", type=str, default="juelich",
                        choices=["juelich", "eth_ucy"], help="数据格式")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[30, 16],
                        help="网格尺寸 (宽 高)")
    parser.add_argument("--target-fps", type=float, default=10.0,
                        help="目标帧率")
    
    args = parser.parse_args()
    
    if args.format == 'juelich':
        convert_juelich_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            grid_size=tuple(args.grid_size),
        )
    else:
        # 通用转换
        converter = TrajectoryToDensityConverter(
            data_format=args.format,
            grid_size=tuple(args.grid_size),
            target_fps=args.target_fps,
        )
        
        input_path = Path(args.input_dir)
        txt_files = sorted(input_path.glob("*.txt"))
        converter.convert_files(txt_files, args.output_dir)
