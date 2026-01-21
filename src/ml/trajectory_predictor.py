"""
Social-LSTM 轨迹预测器
基于 Alahi et al. 2016 "Social LSTM: Human Trajectory Prediction in Crowded Spaces"

核心特点:
- 使用LSTM编码器处理历史轨迹
- Social Pooling层捕捉行人间的交互
- LSTM解码器预测未来轨迹

输入: 过去8帧位置 (8, 2) + 周围行人位置
输出: 未来12帧位置 (12, 2)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class SocialPooling(nn.Module):
    """Social Pooling层 - 捕捉行人间的交互

    将周围2米范围分成grid_size x grid_size网格，
    聚合每个网格内行人的隐藏状态。

    参考: Alahi et al. 2016 Section 3.2
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        pool_dim: int = 64,
        grid_size: int = 4,
        neighborhood_size: float = 2.0
    ):
        """
        Args:
            hidden_dim: LSTM隐藏层维度
            pool_dim: 池化后输出维度
            grid_size: 网格划分大小 (grid_size x grid_size)
            neighborhood_size: 邻域范围 (米)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_dim = pool_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size

        # 网格嵌入层
        self.grid_embedding = nn.Linear(
            hidden_dim * grid_size * grid_size,
            pool_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        seq_start_end: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 所有行人的隐藏状态 (total_peds, hidden_dim)
            positions: 所有行人的当前位置 (total_peds, 2)
            seq_start_end: 每个场景的行人索引范围列表

        Returns:
            social_tensor: 社会池化特征 (total_peds, pool_dim)
        """
        batch_size = hidden_states.size(0)
        social_tensors = []

        for start, end in seq_start_end:
            num_peds = end - start
            if num_peds == 0:
                continue

            # 获取该场景中的隐藏状态和位置
            curr_hidden = hidden_states[start:end]  # (num_peds, hidden_dim)
            curr_pos = positions[start:end]  # (num_peds, 2)

            # 为每个行人计算社会池化张量
            for i in range(num_peds):
                # 以当前行人为中心的网格
                grid_tensor = torch.zeros(
                    self.grid_size * self.grid_size,
                    self.hidden_dim,
                    device=hidden_states.device
                )

                # 计算其他行人相对于当前行人的位置
                rel_pos = curr_pos - curr_pos[i:i+1]  # (num_peds, 2)

                # 确定每个行人落入哪个网格
                for j in range(num_peds):
                    if i == j:
                        continue

                    # 检查是否在邻域范围内
                    if (abs(rel_pos[j, 0]) > self.neighborhood_size or
                        abs(rel_pos[j, 1]) > self.neighborhood_size):
                        continue

                    # 计算网格索引
                    grid_x = int((rel_pos[j, 0] + self.neighborhood_size) /
                               (2 * self.neighborhood_size) * self.grid_size)
                    grid_y = int((rel_pos[j, 1] + self.neighborhood_size) /
                               (2 * self.neighborhood_size) * self.grid_size)

                    # 边界检查
                    grid_x = min(max(grid_x, 0), self.grid_size - 1)
                    grid_y = min(max(grid_y, 0), self.grid_size - 1)

                    grid_idx = grid_y * self.grid_size + grid_x

                    # 累加隐藏状态到对应网格
                    grid_tensor[grid_idx] += curr_hidden[j]

                # 展平并嵌入
                grid_flat = grid_tensor.view(-1)  # (grid_size^2 * hidden_dim)
                social_tensors.append(grid_flat)

        if len(social_tensors) == 0:
            return torch.zeros(batch_size, self.pool_dim, device=hidden_states.device)

        # 堆叠并通过嵌入层
        social_tensor = torch.stack(social_tensors)  # (total_peds, grid_size^2 * hidden_dim)
        social_tensor = self.grid_embedding(social_tensor)  # (total_peds, pool_dim)

        return social_tensor


class SocialLSTM(nn.Module):
    """Social-LSTM轨迹预测模型

    架构:
    1. 位置嵌入: (x, y) -> embedding_dim
    2. LSTM编码器: 处理观测轨迹
    3. Social Pooling: 捕捉行人间交互
    4. LSTM解码器: 生成预测轨迹
    5. 输出层: hidden_dim -> (x, y)

    参考: Alahi et al. 2016
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        pool_dim: int = 64,
        grid_size: int = 4,
        neighborhood_size: float = 2.0,
        dropout: float = 0.0
    ):
        """
        Args:
            obs_len: 观测序列长度 (帧数)
            pred_len: 预测序列长度 (帧数)
            embedding_dim: 位置嵌入维度
            hidden_dim: LSTM隐藏层维度
            pool_dim: Social Pooling输出维度
            grid_size: 社会池化网格大小
            neighborhood_size: 邻域范围 (米)
            dropout: Dropout概率
        """
        super().__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pool_dim = pool_dim

        # 位置嵌入层
        self.spatial_embedding = nn.Linear(2, embedding_dim)

        # LSTM编码器
        self.encoder_lstm = nn.LSTMCell(embedding_dim, hidden_dim)

        # Social Pooling层
        self.social_pool = SocialPooling(
            hidden_dim=hidden_dim,
            pool_dim=pool_dim,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        # LSTM解码器 (输入: embedding + social pooling)
        self.decoder_lstm = nn.LSTMCell(embedding_dim + pool_dim, hidden_dim)

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 2)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化LSTM隐藏状态"""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def encode(
        self,
        obs_traj: torch.Tensor,
        seq_start_end: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码观测轨迹

        Args:
            obs_traj: 观测轨迹 (obs_len, total_peds, 2)
            seq_start_end: 每个场景的行人索引范围

        Returns:
            hidden: 编码后的隐藏状态 (total_peds, hidden_dim)
            cell: 编码后的细胞状态 (total_peds, hidden_dim)
        """
        batch_size = obs_traj.size(1)
        device = obs_traj.device

        # 初始化隐藏状态
        h, c = self.init_hidden(batch_size, device)

        # 逐帧编码
        for t in range(self.obs_len):
            # 位置嵌入
            pos = obs_traj[t]  # (total_peds, 2)
            embedded = self.spatial_embedding(pos)  # (total_peds, embedding_dim)
            embedded = self.dropout(embedded)

            # LSTM更新
            h, c = self.encoder_lstm(embedded, (h, c))

        return h, c

    def decode(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        last_pos: torch.Tensor,
        seq_start_end: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """解码预测轨迹

        Args:
            hidden: 编码器输出的隐藏状态 (total_peds, hidden_dim)
            cell: 编码器输出的细胞状态 (total_peds, hidden_dim)
            last_pos: 最后一个观测位置 (total_peds, 2)
            seq_start_end: 每个场景的行人索引范围

        Returns:
            pred_traj: 预测轨迹 (pred_len, total_peds, 2)
        """
        batch_size = hidden.size(0)
        device = hidden.device

        pred_traj = []
        h, c = hidden, cell
        curr_pos = last_pos

        for t in range(self.pred_len):
            # 位置嵌入
            embedded = self.spatial_embedding(curr_pos)  # (total_peds, embedding_dim)
            embedded = self.dropout(embedded)

            # Social Pooling
            social_tensor = self.social_pool(h, curr_pos, seq_start_end)  # (total_peds, pool_dim)

            # 拼接嵌入和社会池化特征
            decoder_input = torch.cat([embedded, social_tensor], dim=1)  # (total_peds, embedding_dim + pool_dim)

            # LSTM解码
            h, c = self.decoder_lstm(decoder_input, (h, c))

            # 输出位置偏移
            output = self.output_layer(h)  # (total_peds, 2)

            # 累加得到绝对位置
            curr_pos = curr_pos + output
            pred_traj.append(curr_pos)

        pred_traj = torch.stack(pred_traj, dim=0)  # (pred_len, total_peds, 2)
        return pred_traj

    def forward(
        self,
        obs_traj: torch.Tensor,
        seq_start_end: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """前向传播

        Args:
            obs_traj: 观测轨迹 (obs_len, total_peds, 2)
            seq_start_end: 每个场景的行人索引范围

        Returns:
            pred_traj: 预测轨迹 (pred_len, total_peds, 2)
        """
        # 编码
        h, c = self.encode(obs_traj, seq_start_end)

        # 最后一个观测位置
        last_pos = obs_traj[-1]  # (total_peds, 2)

        # 解码
        pred_traj = self.decode(h, c, last_pos, seq_start_end)

        return pred_traj

    @classmethod
    def load(cls, model_path: str, device: str = 'cpu') -> 'SocialLSTM':
        """加载训练好的模型

        Args:
            model_path: 模型文件路径
            device: 设备 ('cpu' 或 'cuda')

        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(model_path, map_location=device)

        # 创建模型实例
        model = cls(
            obs_len=checkpoint.get('obs_len', 8),
            pred_len=checkpoint.get('pred_len', 12),
            embedding_dim=checkpoint.get('embedding_dim', 64),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            pool_dim=checkpoint.get('pool_dim', 64),
            grid_size=checkpoint.get('grid_size', 4),
            neighborhood_size=checkpoint.get('neighborhood_size', 2.0),
            dropout=0.0  # 推理时不用dropout
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"Social-LSTM模型已加载: {model_path}")
        print(f"  观测长度: {model.obs_len}, 预测长度: {model.pred_len}")

        return model

    def save(self, model_path: str):
        """保存模型

        Args:
            model_path: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'obs_len': self.obs_len,
            'pred_len': self.pred_len,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'pool_dim': self.pool_dim,
        }
        torch.save(checkpoint, model_path)
        print(f"模型已保存到: {model_path}")


class TrajectoryPredictor:
    """轨迹预测器封装类

    为仿真系统提供简洁的接口:
    1. 管理历史轨迹缓冲区
    2. 批量预测所有行人的未来轨迹
    3. 角落陷阱检测
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        obs_len: int = 8,
        pred_len: int = 12,
        device: str = 'cpu'
    ):
        """
        Args:
            model_path: Social-LSTM模型路径 (None则使用线性外推)
            obs_len: 观测序列长度
            pred_len: 预测序列长度
            device: 运算设备
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.device = device

        # 历史轨迹缓冲区: {ped_id: [(x, y), ...]}
        self.history_buffer: Dict[int, List[np.ndarray]] = {}

        # 加载神经网络模型
        self.model = None
        if model_path and Path(model_path).exists():
            try:
                self.model = SocialLSTM.load(model_path, device)
                self.use_neural_network = True
                print("轨迹预测器: 使用Social-LSTM神经网络")
            except Exception as e:
                print(f"加载Social-LSTM失败: {e}")
                self.use_neural_network = False
                print("轨迹预测器: 回退到线性外推")
        else:
            self.use_neural_network = False
            print("轨迹预测器: 使用线性外推 (未找到神经网络模型)")

    def update_history(self, ped_id: int, position: np.ndarray):
        """更新行人历史轨迹

        Args:
            ped_id: 行人ID
            position: 当前位置 (x, y)
        """
        if ped_id not in self.history_buffer:
            self.history_buffer[ped_id] = []

        self.history_buffer[ped_id].append(position.copy())

        # 保持固定长度
        if len(self.history_buffer[ped_id]) > self.obs_len:
            self.history_buffer[ped_id].pop(0)

    def remove_pedestrian(self, ped_id: int):
        """移除已疏散的行人"""
        if ped_id in self.history_buffer:
            del self.history_buffer[ped_id]

    def predict_trajectory_linear(
        self,
        ped_id: int,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        dt: float = 0.4
    ) -> np.ndarray:
        """线性外推预测 (回退方案)

        Args:
            ped_id: 行人ID
            current_pos: 当前位置
            current_vel: 当前速度
            dt: 时间步长

        Returns:
            predicted_trajectory: (pred_len, 2) 预测轨迹
        """
        pred_traj = np.zeros((self.pred_len, 2))
        pos = current_pos.copy()

        for t in range(self.pred_len):
            pos = pos + current_vel * dt
            pred_traj[t] = pos

        return pred_traj

    def predict_all_trajectories(
        self,
        pedestrians: List,
        scene_bounds: Tuple[float, float, float, float] = None
    ) -> Dict[int, np.ndarray]:
        """预测所有行人的未来轨迹

        Args:
            pedestrians: 行人列表 (需要有id, position, velocity属性)
            scene_bounds: 场景边界 (x_min, y_min, x_max, y_max)

        Returns:
            predictions: {ped_id: (pred_len, 2)} 预测轨迹字典
        """
        predictions = {}

        if len(pedestrians) == 0:
            return predictions

        # 更新历史缓冲区
        for ped in pedestrians:
            self.update_history(ped.id, ped.position)

        # 如果有神经网络且历史足够，使用神经网络预测
        if self.use_neural_network and self._has_enough_history(pedestrians):
            predictions = self._predict_with_neural_network(pedestrians)
        else:
            # 回退到线性外推
            for ped in pedestrians:
                pred = self.predict_trajectory_linear(
                    ped.id, ped.position, ped.velocity
                )
                predictions[ped.id] = pred

        # 限制在场景边界内
        if scene_bounds:
            x_min, y_min, x_max, y_max = scene_bounds
            for ped_id, traj in predictions.items():
                traj[:, 0] = np.clip(traj[:, 0], x_min, x_max)
                traj[:, 1] = np.clip(traj[:, 1], y_min, y_max)

        return predictions

    def _has_enough_history(self, pedestrians: List) -> bool:
        """检查是否有足够的历史数据用于神经网络预测"""
        for ped in pedestrians:
            if ped.id not in self.history_buffer:
                return False
            if len(self.history_buffer[ped.id]) < self.obs_len:
                return False
        return True

    def _predict_with_neural_network(self, pedestrians: List) -> Dict[int, np.ndarray]:
        """使用Social-LSTM进行批量预测"""
        predictions = {}

        try:
            # 准备输入数据
            obs_traj_list = []
            ped_ids = []

            for ped in pedestrians:
                history = self.history_buffer[ped.id][-self.obs_len:]
                obs_traj_list.append(np.array(history))
                ped_ids.append(ped.id)

            # 转换为PyTorch张量 (obs_len, total_peds, 2)
            obs_traj = np.array(obs_traj_list)  # (total_peds, obs_len, 2)
            obs_traj = np.transpose(obs_traj, (1, 0, 2))  # (obs_len, total_peds, 2)
            obs_traj_tensor = torch.FloatTensor(obs_traj).to(self.device)

            # 所有行人属于同一个场景
            seq_start_end = [(0, len(pedestrians))]

            # 神经网络预测
            with torch.no_grad():
                pred_traj = self.model(obs_traj_tensor, seq_start_end)
                pred_traj = pred_traj.cpu().numpy()  # (pred_len, total_peds, 2)

            # 整理结果
            pred_traj = np.transpose(pred_traj, (1, 0, 2))  # (total_peds, pred_len, 2)
            for i, ped_id in enumerate(ped_ids):
                predictions[ped_id] = pred_traj[i]

        except Exception as e:
            # 神经网络预测失败，回退到线性外推
            print(f"Neural network prediction failed: {e}, falling back to linear")
            for ped in pedestrians:
                pred = self.predict_trajectory_linear(
                    ped.id, ped.position, ped.velocity
                )
                predictions[ped.id] = pred

        return predictions

    def detect_corner_trap(
        self,
        pred_trajectory: np.ndarray,
        corners: List[np.ndarray],
        trap_radius: float = 3.0
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """检测行人是否正在走向死角

        Args:
            pred_trajectory: 预测轨迹 (pred_len, 2)
            corners: 角落位置列表
            trap_radius: 陷阱检测半径

        Returns:
            is_trapped: 是否将陷入角落
            trap_corner: 陷阱角落位置 (如果is_trapped为True)
        """
        for future_pos in pred_trajectory:
            for corner in corners:
                if np.linalg.norm(future_pos - corner) < trap_radius:
                    return True, corner
        return False, None

    def predict_exit_congestion(
        self,
        predictions: Dict[int, np.ndarray],
        exit_positions: List[np.ndarray],
        detection_radius: float = 10.0
    ) -> Dict[int, int]:
        """预测每个出口的拥堵程度

        Args:
            predictions: 预测轨迹字典
            exit_positions: 出口位置列表
            detection_radius: 检测半径

        Returns:
            exit_counts: {exit_idx: 预测人数}
        """
        exit_counts = {i: 0 for i in range(len(exit_positions))}

        for ped_id, traj in predictions.items():
            # 检查轨迹终点最接近哪个出口
            final_pos = traj[-1]
            min_dist = float('inf')
            nearest_exit = 0

            for i, exit_pos in enumerate(exit_positions):
                dist = np.linalg.norm(final_pos - exit_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_exit = i

            if min_dist < detection_radius:
                exit_counts[nearest_exit] += 1

        return exit_counts


def trajectory_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """轨迹预测损失函数: ADE + FDE

    ADE (Average Displacement Error): 平均位移误差
    FDE (Final Displacement Error): 最终位移误差

    Args:
        pred: 预测轨迹 (pred_len, batch, 2)
        target: 真实轨迹 (pred_len, batch, 2)

    Returns:
        loss: ADE + FDE
    """
    # ADE: 所有时间步的平均误差
    ade = torch.mean(torch.norm(pred - target, dim=-1))

    # FDE: 最后一个时间步的误差
    fde = torch.mean(torch.norm(pred[-1] - target[-1], dim=-1))

    return ade + fde


def compute_ade_fde(
    pred: np.ndarray,
    target: np.ndarray
) -> Tuple[float, float]:
    """计算ADE和FDE指标

    Args:
        pred: 预测轨迹 (pred_len, 2) 或 (batch, pred_len, 2)
        target: 真实轨迹 (pred_len, 2) 或 (batch, pred_len, 2)

    Returns:
        ade: 平均位移误差
        fde: 最终位移误差
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    # ADE
    displacements = np.linalg.norm(pred - target, axis=-1)  # (batch, pred_len)
    ade = np.mean(displacements)

    # FDE
    final_displacements = np.linalg.norm(pred[:, -1] - target[:, -1], axis=-1)
    fde = np.mean(final_displacements)

    return ade, fde
