"""
密度场预测主模块

核心功能：
1. 计算当前密度网格
2. 计算流场网格
3. 计算出口距离场
4. 调用ConvLSTM进行密度预测
5. 与动态分流模块集成

设计原则：
- 网格结构固定（30×16），不随人数变化
- 输入/输出密度归一化到[0,1]（基于最大安全密度4.0人/m²）
- 与SFM仿真解耦，支持独立测试

参考文档: docs/new_station_plan.md 6.3节
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from dataclasses import dataclass

from .conv_lstm import DensityPredictorNet, DensityPredictorLite


# ========== 常量定义 ==========
GRID_SIZE = (30, 16)          # 固定网格 (宽度×高度)
CELL_SIZE = 5.0               # 每格5m×5m
MAX_SAFE_DENSITY = 4.0        # 最大安全密度 (人/m²)
MAX_VELOCITY = 2.0            # 最大速度 (m/s)
SCENE_SIZE = (150.0, 80.0)    # 场景尺寸 (m)


@dataclass
class DensityField:
    """密度场数据结构"""
    density: np.ndarray       # [30, 16] 密度网格
    flow_x: np.ndarray        # [30, 16] X方向流场
    flow_y: np.ndarray        # [30, 16] Y方向流场
    exit_distance: np.ndarray # [30, 16] 出口距离场
    timestamp: float          # 时间戳
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """转换为模型输入张量 [4, 30, 16]"""
        data = np.stack([
            self.density,
            self.flow_x,
            self.flow_y,
            self.exit_distance,
        ], axis=0)
        return torch.from_numpy(data).float().to(device)


class DensityFieldPredictor:
    """密度场预测器
    
    主要功能：
    1. 从行人位置计算密度网格
    2. 从行人速度计算流场网格
    3. 预计算出口距离场
    4. 使用ConvLSTM预测未来密度
    
    使用示例:
        predictor = DensityFieldPredictor(exits=exits)
        
        # 收集当前帧
        field = predictor.compute_density_field(pedestrians)
        predictor.add_frame(field)
        
        # 预测未来密度
        if predictor.has_enough_frames():
            future_density = predictor.predict()
    """
    
    def __init__(
        self,
        exits: List[Dict],
        grid_size: Tuple[int, int] = GRID_SIZE,
        cell_size: float = CELL_SIZE,
        scene_size: Tuple[float, float] = SCENE_SIZE,
        max_safe_density: float = MAX_SAFE_DENSITY,
        max_velocity: float = MAX_VELOCITY,
        history_length: int = 10,
        prediction_horizon: float = 5.0,
        model_path: Optional[str] = None,
        use_lite_model: bool = False,
        device: str = 'auto',
    ):
        """
        Args:
            exits: 出口列表 [{'id': str, 'position': np.ndarray}, ...]
            grid_size: 网格尺寸 (width, height)
            cell_size: 单元格大小 (m)
            scene_size: 场景尺寸 (m)
            max_safe_density: 最大安全密度 (人/m²)
            max_velocity: 最大速度 (m/s)
            history_length: 历史帧数（10帧 = 1秒，dt=0.1s）
            prediction_horizon: 预测时间窗口 (秒)
            model_path: 预训练模型路径
            use_lite_model: 是否使用轻量级模型
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.scene_size = scene_size
        self.max_safe_density = max_safe_density
        self.max_velocity = max_velocity
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        
        # 出口位置
        self.exits = exits
        
        # 设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 预计算出口距离场
        self.exit_distance_field = self._compute_exit_distance_field()
        
        # 历史帧缓存
        self.history_frames: List[DensityField] = []
        
        # 加载模型
        self.use_lite_model = use_lite_model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: Optional[str]) -> torch.nn.Module:
        """加载预测模型"""
        if self.use_lite_model:
            model = DensityPredictorLite(
                input_channels=4,
                hidden_channels=32,
                grid_size=self.grid_size,
            )
        else:
            model = DensityPredictorNet(
                input_channels=4,
                hidden_channels=64,
                encoder_channels=32,
                kernel_size=3,
                num_lstm_layers=2,
                grid_size=self.grid_size,
            )
        
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"[DensityPredictor] 加载模型: {model_path}")
        else:
            print(f"[DensityPredictor] 使用未训练模型 (设备: {self.device})")
            
        model = model.to(self.device)
        model.eval()
        return model
    
    def _compute_exit_distance_field(self) -> np.ndarray:
        """预计算出口距离场（归一化）"""
        grid_w, grid_h = self.grid_size
        distance_field = np.full((grid_w, grid_h), np.inf)
        
        # 计算每个网格到最近出口的距离
        for i in range(grid_w):
            for j in range(grid_h):
                # 网格中心坐标
                x = (i + 0.5) * self.cell_size
                y = (j + 0.5) * self.cell_size
                cell_pos = np.array([x, y])
                
                # 找最近出口
                min_dist = np.inf
                for exit_info in self.exits:
                    exit_pos = exit_info.get('position', exit_info.get('pos', np.zeros(2)))
                    if isinstance(exit_pos, (list, tuple)):
                        exit_pos = np.array(exit_pos)
                    dist = np.linalg.norm(cell_pos - exit_pos)
                    if dist < min_dist:
                        min_dist = dist
                
                distance_field[i, j] = min_dist
        
        # 归一化到[0, 1]
        max_dist = np.max(distance_field[np.isfinite(distance_field)])
        if max_dist > 0:
            distance_field = distance_field / max_dist
        distance_field = np.clip(distance_field, 0.0, 1.0)
        
        return distance_field
    
    def _pos_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """将场景坐标转换为网格坐标"""
        i = int(x / self.cell_size)
        j = int(y / self.cell_size)
        i = max(0, min(i, self.grid_size[0] - 1))
        j = max(0, min(j, self.grid_size[1] - 1))
        return i, j
    
    def _grid_to_pos(self, i: int, j: int) -> Tuple[float, float]:
        """将网格坐标转换为场景坐标（网格中心）"""
        x = (i + 0.5) * self.cell_size
        y = (j + 0.5) * self.cell_size
        return x, y
    
    def compute_density_field(
        self,
        pedestrians: List[Dict],
        timestamp: float = 0.0,
    ) -> DensityField:
        """计算当前密度场
        
        Args:
            pedestrians: 行人列表 [{'position': np.ndarray, 'velocity': np.ndarray}, ...]
            timestamp: 时间戳
            
        Returns:
            DensityField: 密度场数据
        """
        grid_w, grid_h = self.grid_size
        
        # 初始化网格
        density_grid = np.zeros((grid_w, grid_h))
        flow_x_grid = np.zeros((grid_w, grid_h))
        flow_y_grid = np.zeros((grid_w, grid_h))
        count_grid = np.zeros((grid_w, grid_h))
        
        # 统计每个网格的行人
        for ped in pedestrians:
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
        cell_area = self.cell_size ** 2  # m²
        
        for i in range(grid_w):
            for j in range(grid_h):
                count = count_grid[i, j]
                if count > 0:
                    # 密度 = 人数 / 面积 (人/m²)
                    local_density = count / cell_area
                    # 归一化到[0, 1]
                    density_grid[i, j] = min(local_density / self.max_safe_density, 1.0)
                    # 平均流场
                    flow_x_grid[i, j] = (flow_x_grid[i, j] / count) / self.max_velocity
                    flow_y_grid[i, j] = (flow_y_grid[i, j] / count) / self.max_velocity
                    
        # 流场归一化到[-1, 1]后clip到[-0.5, 0.5]再映射到[0, 1]
        flow_x_grid = np.clip(flow_x_grid, -1.0, 1.0) * 0.5 + 0.5
        flow_y_grid = np.clip(flow_y_grid, -1.0, 1.0) * 0.5 + 0.5
        
        return DensityField(
            density=density_grid,
            flow_x=flow_x_grid,
            flow_y=flow_y_grid,
            exit_distance=self.exit_distance_field.copy(),
            timestamp=timestamp,
        )
    
    def add_frame(self, field: DensityField):
        """添加一帧到历史缓存"""
        self.history_frames.append(field)
        
        # 保持固定长度
        if len(self.history_frames) > self.history_length:
            self.history_frames.pop(0)
    
    def has_enough_frames(self) -> bool:
        """是否有足够的历史帧用于预测"""
        return len(self.history_frames) >= self.history_length
    
    def clear_history(self):
        """清空历史帧"""
        self.history_frames.clear()
    
    @torch.no_grad()
    def predict(self) -> np.ndarray:
        """预测未来密度场
        
        Returns:
            predicted_density: [30, 16] 预测密度网格（归一化）
        """
        if not self.has_enough_frames():
            # 如果历史帧不足，返回最后一帧的密度
            if self.history_frames:
                return self.history_frames[-1].density.copy()
            else:
                return np.zeros(self.grid_size)
        
        # 构建输入张量
        frames = [f.to_tensor(self.device) for f in self.history_frames[-self.history_length:]]
        x = torch.stack(frames, dim=0).unsqueeze(0)  # [1, seq_len, 4, h, w]
        
        # 模型推理
        self.model.eval()
        prediction, _ = self.model(x)
        
        # 转换为numpy
        pred_density = prediction.squeeze().cpu().numpy()
        
        return pred_density
    
    @torch.no_grad()
    def predict_multi_step(self, steps: int = 5) -> List[np.ndarray]:
        """多步预测
        
        Args:
            steps: 预测步数
            
        Returns:
            predictions: 预测密度序列 [[30, 16], ...]
        """
        if not self.has_enough_frames():
            if self.history_frames:
                return [self.history_frames[-1].density.copy()] * steps
            else:
                return [np.zeros(self.grid_size)] * steps
        
        # 构建输入张量
        frames = [f.to_tensor(self.device) for f in self.history_frames[-self.history_length:]]
        x = torch.stack(frames, dim=0).unsqueeze(0)
        
        # 多步预测
        self.model.eval()
        predictions = self.model.predict_multi_step(x, steps=steps)
        
        # 转换为numpy列表
        result = []
        for i in range(steps):
            pred = predictions[0, i, 0].cpu().numpy()
            result.append(pred)
        
        return result
    
    def get_exit_predicted_densities(
        self,
        predicted_density: Optional[np.ndarray] = None,
        radius: int = 2,
    ) -> Dict[str, float]:
        """获取各出口的预测密度
        
        Args:
            predicted_density: 预测密度网格，如果为None则调用predict()
            radius: 出口周围的网格半径
            
        Returns:
            {exit_id: predicted_density}
        """
        if predicted_density is None:
            predicted_density = self.predict()
        
        result = {}
        for exit_info in self.exits:
            exit_id = exit_info.get('id', 'unknown')
            exit_pos = exit_info.get('position', exit_info.get('pos', np.zeros(2)))
            
            if isinstance(exit_pos, (list, tuple)):
                exit_pos = np.array(exit_pos)
            
            # 出口所在网格
            i, j = self._pos_to_grid(exit_pos[0], exit_pos[1])
            
            # 计算周围区域的平均密度
            densities = []
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                        densities.append(predicted_density[ni, nj])
            
            if densities:
                result[exit_id] = float(np.mean(densities))
            else:
                result[exit_id] = 0.0
        
        return result
    
    def get_high_density_regions(
        self,
        predicted_density: Optional[np.ndarray] = None,
        threshold: float = 0.75,
    ) -> List[Tuple[int, int, float]]:
        """获取高密度区域
        
        Args:
            predicted_density: 预测密度网格
            threshold: 密度阈值 (归一化值)
            
        Returns:
            [(i, j, density), ...] 高密度网格列表
        """
        if predicted_density is None:
            predicted_density = self.predict()
        
        high_density_cells = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if predicted_density[i, j] > threshold:
                    high_density_cells.append((i, j, predicted_density[i, j]))
        
        # 按密度排序
        high_density_cells.sort(key=lambda x: x[2], reverse=True)
        
        return high_density_cells
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.history_frames:
            return {
                'history_length': 0,
                'has_enough_frames': False,
                'device': self.device,
                'model_type': 'lite' if self.use_lite_model else 'full',
            }
        
        latest = self.history_frames[-1]
        return {
            'history_length': len(self.history_frames),
            'has_enough_frames': self.has_enough_frames(),
            'latest_timestamp': latest.timestamp,
            'max_current_density': float(np.max(latest.density)),
            'mean_current_density': float(np.mean(latest.density)),
            'device': self.device,
            'model_type': 'lite' if self.use_lite_model else 'full',
        }


def create_predictor_from_env(env, model_path: Optional[str] = None) -> DensityFieldPredictor:
    """从环境创建密度预测器
    
    Args:
        env: LargeStationEnv 环境实例
        model_path: 模型路径
        
    Returns:
        DensityFieldPredictor 实例
    """
    # 提取出口信息
    exits = []
    for exit_obj in env.exits:
        exits.append({
            'id': exit_obj.id,
            'position': exit_obj.position.copy(),
        })
    
    return DensityFieldPredictor(
        exits=exits,
        model_path=model_path,
    )
