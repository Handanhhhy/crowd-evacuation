"""
GPU加速社会力模型
使用PyTorch实现，支持CUDA/MPS/CPU自动切换

核心优化:
- 所有行人状态存储为PyTorch张量
- 距离矩阵批量计算（向量化O(n²) -> GPU并行）
- 社会力、障碍物力批量计算
- 支持MPS (Apple Silicon) / CUDA / CPU自动切换

预期加速比:
| 行人数 | CPU时间 | GPU时间 | 加速比 |
|--------|---------|---------|--------|
| 80人   | 1x      | 0.5x    | 2倍    |
| 500人  | 40x     | 1x      | 40倍   |
| 1000人 | 150x    | 1.5x    | 100倍  |

文献参考:
- Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


class PedestrianType(Enum):
    """行人类型枚举"""
    NORMAL = "normal"
    ELDERLY = "elderly"
    CHILD = "child"
    IMPATIENT = "impatient"
    WITH_SMALL_BAG = "with_small_bag"      # 携带小包
    WITH_LUGGAGE = "with_luggage"          # 携带拉杆箱
    WITH_LARGE_LUGGAGE = "with_large_luggage"  # 携带大行李


# 行人类型参数配置 (基于文献)
PEDESTRIAN_TYPE_PARAMS = {
    PedestrianType.NORMAL: {
        'desired_speed': 1.34,
        'speed_std': 0.26,
        'reaction_time': 0.5,
        'radius': 0.3,
        'color': 'blue',
    },
    PedestrianType.ELDERLY: {
        'desired_speed': 0.9,
        'speed_std': 0.15,
        'reaction_time': 0.8,
        'radius': 0.3,
        'color': 'green',
    },
    PedestrianType.CHILD: {
        'desired_speed': 0.7,
        'speed_std': 0.2,
        'reaction_time': 0.6,
        'radius': 0.25,
        'color': 'yellow',
    },
    PedestrianType.IMPATIENT: {
        'desired_speed': 1.6,
        'speed_std': 0.2,
        'reaction_time': 0.3,
        'radius': 0.3,
        'color': 'red',
    },
    PedestrianType.WITH_SMALL_BAG: {
        'desired_speed': 1.2,
        'speed_std': 0.2,
        'reaction_time': 0.5,
        'radius': 0.35,
        'color': 'cyan',
    },
    PedestrianType.WITH_LUGGAGE: {
        'desired_speed': 0.9,
        'speed_std': 0.15,
        'reaction_time': 0.6,
        'radius': 0.5,
        'color': 'orange',
    },
    PedestrianType.WITH_LARGE_LUGGAGE: {
        'desired_speed': 0.7,
        'speed_std': 0.1,
        'reaction_time': 0.7,
        'radius': 0.6,
        'color': 'purple',
    },
}


@dataclass
class Pedestrian:
    """行人状态（CPU兼容版本，用于接口）"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    target: np.ndarray
    desired_speed: float = 1.34
    radius: float = 0.3
    ped_type: PedestrianType = PedestrianType.NORMAL
    reaction_time: float = 0.5
    is_waiting: bool = False
    wait_timer: float = 0.0
    panic_factor: float = 0.0
    stuck_timer: float = 0.0
    last_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    guidance_count: int = 0
    last_guidance_time: float = -999.0
    original_target: Optional[np.ndarray] = field(default=None)

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

    @classmethod
    def create_with_type(
        cls,
        id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        target: np.ndarray,
        ped_type: PedestrianType = PedestrianType.NORMAL,
        speed_variation: bool = True
    ) -> 'Pedestrian':
        """根据类型创建行人"""
        params = PEDESTRIAN_TYPE_PARAMS[ped_type]

        if speed_variation:
            desired_speed = np.random.normal(
                params['desired_speed'],
                params['speed_std']
            )
            desired_speed = np.clip(desired_speed, 0.3, 2.0)
        else:
            desired_speed = params['desired_speed']

        return cls(
            id=id,
            position=position,
            velocity=velocity,
            target=target,
            desired_speed=desired_speed,
            radius=params['radius'],
            ped_type=ped_type,
            reaction_time=params['reaction_time'],
            is_waiting=False,
            wait_timer=0.0,
            panic_factor=0.0,
            stuck_timer=0.0,
            last_position=position.copy(),
            guidance_count=0,
            last_guidance_time=-999.0,
            original_target=target.copy()
        )


class GPUSocialForceModel:
    """GPU加速的社会力模型

    F_total = F_drive + F_social + F_obstacle + F_random

    核心特性:
    - 使用PyTorch张量存储所有行人状态
    - 批量计算距离矩阵和力
    - 支持CUDA/MPS/CPU自动切换
    """

    def __init__(
        self,
        device: str = 'auto',
        tau: float = 0.5,
        A: float = 2000.0,
        B: float = 0.08,
        k: float = 1.2e5,
        kappa: float = 2.4e5,
        wall_A: float = 2000.0,
        wall_B: float = 0.08,
        enable_waiting: bool = True,
        waiting_density_threshold: float = 0.8,
        enable_perturbation: bool = True,
        perturbation_sigma: float = 0.1,
        enable_panic: bool = True,
        panic_density_threshold: float = 1.5,
        gbm_predictor=None,
        gbm_weight: float = 0.3,
    ):
        # 设备选择
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"GPU SFM使用设备: CUDA ({torch.cuda.get_device_name(0)})")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("GPU SFM使用设备: MPS (Apple Silicon)")
            else:
                self.device = torch.device('cpu')
                print("GPU SFM使用设备: CPU")
        else:
            self.device = torch.device(device)
            print(f"GPU SFM使用设备: {device}")

        # SFM参数
        self.tau = tau
        self.A = A
        self.B = B
        self.k = k
        self.kappa = kappa
        self.wall_A = wall_A
        self.wall_B = wall_B

        # 行为参数
        self.enable_waiting = enable_waiting
        self.waiting_density_threshold = waiting_density_threshold
        self.enable_perturbation = enable_perturbation
        self.perturbation_sigma = perturbation_sigma
        self.enable_panic = enable_panic
        self.panic_density_threshold = panic_density_threshold

        # GBM预测器
        self.gbm_predictor = gbm_predictor
        self.gbm_weight = gbm_weight

        # 行人列表（CPU端，用于接口兼容）
        self.pedestrians: List[Pedestrian] = []

        # 障碍物列表（线段）
        self.obstacles: List[np.ndarray] = []

        # GPU张量缓存
        self._positions_tensor: Optional[torch.Tensor] = None
        self._velocities_tensor: Optional[torch.Tensor] = None
        self._targets_tensor: Optional[torch.Tensor] = None
        self._radii_tensor: Optional[torch.Tensor] = None
        self._desired_speeds_tensor: Optional[torch.Tensor] = None
        self._reaction_times_tensor: Optional[torch.Tensor] = None
        self._panic_factors_tensor: Optional[torch.Tensor] = None
        self._is_waiting_tensor: Optional[torch.Tensor] = None

        # 障碍物GPU张量
        self._obstacles_start_tensor: Optional[torch.Tensor] = None
        self._obstacles_end_tensor: Optional[torch.Tensor] = None

    def add_pedestrian(self, ped: Pedestrian) -> None:
        """添加行人"""
        self.pedestrians.append(ped)
        self._invalidate_cache()

    def add_obstacle(self, start: np.ndarray, end: np.ndarray) -> None:
        """添加障碍物（线段）"""
        self.obstacles.append(np.array([start, end]))
        self._invalidate_obstacle_cache()

    def _invalidate_cache(self):
        """使缓存失效"""
        self._positions_tensor = None
        self._velocities_tensor = None
        self._targets_tensor = None

    def _invalidate_obstacle_cache(self):
        """使障碍物缓存失效"""
        self._obstacles_start_tensor = None
        self._obstacles_end_tensor = None

    def _sync_to_gpu(self):
        """将行人数据同步到GPU"""
        n = len(self.pedestrians)
        if n == 0:
            return

        # 批量提取数据
        positions = np.array([p.position for p in self.pedestrians])
        velocities = np.array([p.velocity for p in self.pedestrians])
        targets = np.array([p.target for p in self.pedestrians])
        radii = np.array([p.radius for p in self.pedestrians])
        desired_speeds = np.array([p.desired_speed for p in self.pedestrians])
        reaction_times = np.array([p.reaction_time for p in self.pedestrians])
        panic_factors = np.array([p.panic_factor for p in self.pedestrians])
        is_waiting = np.array([p.is_waiting for p in self.pedestrians])

        # 转换为GPU张量
        self._positions_tensor = torch.tensor(positions, device=self.device, dtype=torch.float32)
        self._velocities_tensor = torch.tensor(velocities, device=self.device, dtype=torch.float32)
        self._targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float32)
        self._radii_tensor = torch.tensor(radii, device=self.device, dtype=torch.float32)
        self._desired_speeds_tensor = torch.tensor(desired_speeds, device=self.device, dtype=torch.float32)
        self._reaction_times_tensor = torch.tensor(reaction_times, device=self.device, dtype=torch.float32)
        self._panic_factors_tensor = torch.tensor(panic_factors, device=self.device, dtype=torch.float32)
        self._is_waiting_tensor = torch.tensor(is_waiting, device=self.device, dtype=torch.bool)

    def _sync_obstacles_to_gpu(self):
        """将障碍物数据同步到GPU"""
        if len(self.obstacles) == 0:
            self._obstacles_start_tensor = torch.zeros((0, 2), device=self.device, dtype=torch.float32)
            self._obstacles_end_tensor = torch.zeros((0, 2), device=self.device, dtype=torch.float32)
            return

        starts = np.array([obs[0] for obs in self.obstacles])
        ends = np.array([obs[1] for obs in self.obstacles])

        self._obstacles_start_tensor = torch.tensor(starts, device=self.device, dtype=torch.float32)
        self._obstacles_end_tensor = torch.tensor(ends, device=self.device, dtype=torch.float32)

    def _sync_from_gpu(self):
        """将GPU数据同步回CPU"""
        if self._positions_tensor is None:
            return

        positions = self._positions_tensor.cpu().numpy()
        velocities = self._velocities_tensor.cpu().numpy()

        for i, ped in enumerate(self.pedestrians):
            ped.last_position = ped.position.copy()
            ped.position = positions[i].copy()
            ped.velocity = velocities[i].copy()

    def compute_driving_force_gpu(self) -> torch.Tensor:
        """GPU加速计算驱动力 F_drive = (v0 * e - v) / tau

        Returns:
            (n, 2) 驱动力张量
        """
        n = len(self.pedestrians)
        if n == 0:
            return torch.zeros((0, 2), device=self.device, dtype=torch.float32)

        # 计算方向向量
        direction = self._targets_tensor - self._positions_tensor  # (n, 2)
        distance = torch.norm(direction, dim=1, keepdim=True)  # (n, 1)

        # 避免除以零
        distance = torch.clamp(distance, min=1e-6)
        e = direction / distance  # (n, 2) 单位方向

        # 有效期望速度（考虑恐慌因子）
        effective_speed = self._desired_speeds_tensor * (1.0 + self._panic_factors_tensor)
        effective_speed = torch.clamp(effective_speed, max=2.5)  # (n,)

        # 期望速度向量
        desired_velocity = effective_speed.unsqueeze(1) * e  # (n, 2)

        # 驱动力
        force = (desired_velocity - self._velocities_tensor) / self._reaction_times_tensor.unsqueeze(1)

        # 处理等待状态的行人
        waiting_mask = self._is_waiting_tensor.unsqueeze(1)  # (n, 1)
        waiting_force = -self._velocities_tensor / self._reaction_times_tensor.unsqueeze(1)
        force = torch.where(waiting_mask, waiting_force, force)

        # 到达目标的行人不再有驱动力
        arrived_mask = (distance < 0.1).squeeze(1)  # (n,)
        force[arrived_mask] = 0.0

        return force

    def compute_social_force_gpu(self) -> torch.Tensor:
        """GPU加速计算社会力（行人之间的排斥力）

        核心优化: O(n²) -> GPU并行矩阵运算

        F_social = A * exp((r_ij - d_ij) / B) * n_ij

        Returns:
            (n, 2) 社会力张量
        """
        n = len(self.pedestrians)
        if n <= 1:
            return torch.zeros((n, 2), device=self.device, dtype=torch.float32)

        pos = self._positions_tensor  # (n, 2)
        radii = self._radii_tensor  # (n,)
        vel = self._velocities_tensor  # (n, 2)

        # 计算成对差异向量 (n, n, 2)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # pos[i] - pos[j]

        # 计算距离矩阵 (n, n)
        distances = torch.norm(diff, dim=2)

        # 避免除以零，设置对角线为大值
        distances = distances + torch.eye(n, device=self.device) * 1e6
        distances = torch.clamp(distances, min=1e-6)

        # 单位向量 n_ij (n, n, 2)
        n_ij = diff / distances.unsqueeze(2)

        # 半径和矩阵 (n, n)
        radii_sum = radii.unsqueeze(1) + radii.unsqueeze(0)

        # 排斥力指数项 (n, n)
        exp_term = torch.exp((radii_sum - distances) / self.B)

        # 排斥力 (n, n, 2)
        repulsion = self.A * exp_term.unsqueeze(2) * n_ij

        # 身体力（接触时）
        contact_mask = distances < radii_sum  # (n, n)
        body_force_magnitude = self.k * (radii_sum - distances)  # (n, n)
        body_force_magnitude = torch.where(contact_mask, body_force_magnitude, torch.zeros_like(body_force_magnitude))
        body_force = body_force_magnitude.unsqueeze(2) * n_ij  # (n, n, 2)

        # 摩擦力（接触时）
        t_ij = torch.stack([-n_ij[:, :, 1], n_ij[:, :, 0]], dim=2)  # 切向向量 (n, n, 2)
        delta_v_tangent = ((vel.unsqueeze(0) - vel.unsqueeze(1)) * t_ij).sum(dim=2)  # (n, n)
        friction_magnitude = self.kappa * (radii_sum - distances) * delta_v_tangent
        friction_magnitude = torch.where(contact_mask, friction_magnitude, torch.zeros_like(friction_magnitude))
        friction_force = friction_magnitude.unsqueeze(2) * t_ij  # (n, n, 2)

        # 总社会力 = 排斥力 + 身体力 + 摩擦力
        total_social_force = repulsion + body_force + friction_force

        # 对角线置零（自身不产生力）
        mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
        total_social_force = total_social_force * mask.unsqueeze(2)

        # 对每个行人求和 (n, 2)
        return total_social_force.sum(dim=1)

    def compute_obstacle_force_gpu(self) -> torch.Tensor:
        """GPU加速计算障碍物排斥力

        Returns:
            (n, 2) 障碍物力张量
        """
        n = len(self.pedestrians)
        m = len(self.obstacles)

        if n == 0:
            return torch.zeros((0, 2), device=self.device, dtype=torch.float32)

        if m == 0:
            return torch.zeros((n, 2), device=self.device, dtype=torch.float32)

        # 确保障碍物在GPU
        if self._obstacles_start_tensor is None:
            self._sync_obstacles_to_gpu()

        pos = self._positions_tensor  # (n, 2)
        radii = self._radii_tensor  # (n,)
        starts = self._obstacles_start_tensor  # (m, 2)
        ends = self._obstacles_end_tensor  # (m, 2)

        # 计算每个行人到每条线段的最近点
        # 广播: pos (n, 1, 2), starts (1, m, 2), ends (1, m, 2)
        pos_expanded = pos.unsqueeze(1)  # (n, 1, 2)
        starts_expanded = starts.unsqueeze(0)  # (1, m, 2)
        ends_expanded = ends.unsqueeze(0)  # (1, m, 2)

        # 线段向量
        segment = ends_expanded - starts_expanded  # (1, m, 2)
        length_sq = (segment ** 2).sum(dim=2, keepdim=True)  # (1, m, 1)
        length_sq = torch.clamp(length_sq, min=1e-6)

        # 投影参数 t
        to_point = pos_expanded - starts_expanded  # (n, m, 2)
        t = (to_point * segment).sum(dim=2, keepdim=True) / length_sq  # (n, m, 1)
        t = torch.clamp(t, min=0.0, max=1.0)

        # 最近点
        closest = starts_expanded + t * segment  # (n, m, 2)

        # 差异向量和距离
        diff = pos_expanded - closest  # (n, m, 2)
        distances = torch.norm(diff, dim=2)  # (n, m)
        distances = torch.clamp(distances, min=1e-6)

        # 单位向量
        n_vec = diff / distances.unsqueeze(2)  # (n, m, 2)

        # 排斥力
        radii_expanded = radii.unsqueeze(1)  # (n, 1)
        exp_term = torch.exp((radii_expanded - distances) / self.wall_B)  # (n, m)
        repulsion = self.wall_A * exp_term.unsqueeze(2) * n_vec  # (n, m, 2)

        # 接触力
        contact_mask = distances < radii_expanded  # (n, m)
        contact_force_mag = self.k * (radii_expanded - distances)
        contact_force_mag = torch.where(contact_mask, contact_force_mag, torch.zeros_like(contact_force_mag))
        contact_force = contact_force_mag.unsqueeze(2) * n_vec  # (n, m, 2)

        # 近距离额外推力
        close_mask = distances < radii_expanded * 1.5  # (n, m)
        extra_force_mag = 3000 * (1 - distances / (radii_expanded * 1.5))
        extra_force_mag = torch.where(close_mask, extra_force_mag, torch.zeros_like(extra_force_mag))
        extra_force = extra_force_mag.unsqueeze(2) * n_vec  # (n, m, 2)

        # 总障碍物力
        total_force = repulsion + contact_force + extra_force

        # 对所有障碍物求和 (n, 2)
        return total_force.sum(dim=1)

    def compute_perturbation_gpu(self) -> torch.Tensor:
        """GPU加速计算随机扰动力

        Returns:
            (n, 2) 扰动力张量
        """
        n = len(self.pedestrians)
        if n == 0 or not self.enable_perturbation:
            return torch.zeros((n, 2), device=self.device, dtype=torch.float32)

        return torch.randn((n, 2), device=self.device, dtype=torch.float32) * self.perturbation_sigma

    def compute_local_density_gpu(self, radius: float = 2.0) -> torch.Tensor:
        """GPU加速计算局部密度

        Args:
            radius: 检测半径

        Returns:
            (n,) 密度张量
        """
        n = len(self.pedestrians)
        if n <= 1:
            return torch.zeros(n, device=self.device, dtype=torch.float32)

        pos = self._positions_tensor  # (n, 2)

        # 计算距离矩阵
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (n, n, 2)
        distances = torch.norm(diff, dim=2)  # (n, n)

        # 设置对角线为大值（排除自身）
        distances = distances + torch.eye(n, device=self.device) * 1e6

        # 统计半径内的行人数
        count = (distances < radius).sum(dim=1).float()  # (n,)

        # 密度 = 人数 / 面积
        area = np.pi * radius ** 2
        return count / area

    def compute_local_density(self, ped: Pedestrian, radius: float = 2.0) -> float:
        """计算单个行人的局部密度（CPU版本，接口兼容）"""
        count = 0
        for other in self.pedestrians:
            if other.id == ped.id:
                continue
            distance = np.linalg.norm(ped.position - other.position)
            if distance < radius:
                count += 1
        area = np.pi * radius ** 2
        return count / area

    def update_behavior_states_gpu(self, dt: float):
        """GPU加速更新行为状态（等待、恐慌）"""
        n = len(self.pedestrians)
        if n == 0:
            return

        # 计算局部密度
        densities = self.compute_local_density_gpu(radius=2.0)

        # 更新恐慌因子
        if self.enable_panic:
            panic_mask = densities > self.panic_density_threshold
            new_panic = torch.clamp(
                (densities - self.panic_density_threshold) * 0.3,
                max=0.5
            )
            calm_down = torch.clamp(self._panic_factors_tensor - 0.02, min=0.0)
            self._panic_factors_tensor = torch.where(panic_mask, new_panic, calm_down)

        # 同步回CPU
        panic_factors = self._panic_factors_tensor.cpu().numpy()
        for i, ped in enumerate(self.pedestrians):
            ped.panic_factor = float(panic_factors[i])

    def step(self, dt: float = 0.1) -> None:
        """执行一步仿真（GPU加速版本）

        Args:
            dt: 时间步长
        """
        n = len(self.pedestrians)
        if n == 0:
            return

        # 同步数据到GPU
        self._sync_to_gpu()

        # 更新行为状态
        self.update_behavior_states_gpu(dt)

        # 计算各种力（全部GPU并行）
        f_drive = self.compute_driving_force_gpu()  # (n, 2)
        f_social = self.compute_social_force_gpu()  # (n, 2)
        f_obstacle = self.compute_obstacle_force_gpu()  # (n, 2)
        f_random = self.compute_perturbation_gpu()  # (n, 2)

        # 总力
        total_force = f_drive + f_social + f_obstacle + f_random  # (n, 2)

        # 更新速度
        new_velocity = self._velocities_tensor + total_force * dt

        # 限制最大速度
        speed = torch.norm(new_velocity, dim=1, keepdim=True)  # (n, 1)
        max_speed = self._desired_speeds_tensor * (1.5 + self._panic_factors_tensor * 0.3)
        max_speed = max_speed.unsqueeze(1)  # (n, 1)

        # 如果超速则归一化
        over_speed_mask = speed > max_speed
        scale = torch.where(over_speed_mask, max_speed / speed, torch.ones_like(speed))
        new_velocity = new_velocity * scale

        self._velocities_tensor = new_velocity

        # 更新位置
        new_position = self._positions_tensor + self._velocities_tensor * dt
        self._positions_tensor = new_position

        # 同步回CPU
        self._sync_from_gpu()

        # 处理反堵塞（在CPU上处理，因为涉及复杂逻辑）
        self._handle_anti_stuck(dt)

    def _handle_anti_stuck(self, dt: float):
        """处理反堵塞机制（CPU端）"""
        for ped in self.pedestrians:
            actual_movement = np.linalg.norm(ped.position - ped.last_position)
            direction = ped.target - ped.position
            dist_to_target = np.linalg.norm(direction)

            is_stuck = actual_movement < 0.05 * dt and not ped.is_waiting and dist_to_target > 1.0

            if is_stuck:
                ped.stuck_timer += dt
            else:
                ped.stuck_timer = max(0, ped.stuck_timer - dt * 2)

            if ped.stuck_timer > 0.2 and dist_to_target > 0.5:
                stuck_level = min(ped.stuck_timer / 1.5, 1.0)

                if ped.stuck_timer < 0.8:
                    # 轻度卡住 - 随机扰动
                    perturbation = np.random.uniform(-0.8, 0.8, 2)
                    ped.velocity += perturbation * (1 + stuck_level)
                    if dist_to_target > 0.1:
                        ped.velocity += 0.5 * direction / dist_to_target
                elif ped.stuck_timer < 1.5:
                    # 中度卡住 - 横向逃逸
                    if dist_to_target > 0.1:
                        escape_dir = np.array([-direction[1], direction[0]]) / dist_to_target
                        if np.random.random() > 0.5:
                            escape_dir = -escape_dir
                        ped.velocity += escape_dir * 1.2
                        ped.velocity += 0.6 * direction / dist_to_target
                else:
                    # 严重卡住 - 强制推开
                    strong_push = np.random.uniform(-1.5, 1.5, 2)
                    ped.velocity += strong_push
                    if dist_to_target > 0.1:
                        ped.velocity += 0.8 * direction / dist_to_target

    def get_state(self) -> np.ndarray:
        """获取所有行人的状态矩阵

        Returns:
            shape: (n_pedestrians, 6)
            columns: [px, py, vx, vy, target_x, target_y]
        """
        states = []
        for ped in self.pedestrians:
            state = np.concatenate([
                ped.position,
                ped.velocity,
                ped.target
            ])
            states.append(state)
        return np.array(states) if states else np.zeros((0, 6))

    def is_finished(self, threshold: float = 1.0) -> bool:
        """检查是否所有行人都到达目标"""
        for ped in self.pedestrians:
            distance = np.linalg.norm(ped.target - ped.position)
            if distance > threshold:
                return False
        return True


def create_random_pedestrians(
    n: int,
    spawn_area: Tuple[float, float, float, float],
    target: np.ndarray,
    seed: int = None,
    type_distribution: Optional[Dict] = None
) -> List[Pedestrian]:
    """创建随机分布的行人"""
    if seed is not None:
        np.random.seed(seed)

    if type_distribution is None:
        type_distribution = {
            PedestrianType.NORMAL: 0.70,
            PedestrianType.ELDERLY: 0.15,
            PedestrianType.CHILD: 0.10,
            PedestrianType.IMPATIENT: 0.05,
        }

    types = list(type_distribution.keys())
    probs = list(type_distribution.values())
    total = sum(probs)
    probs = [p / total for p in probs]

    pedestrians = []
    x_min, y_min, x_max, y_max = spawn_area

    for i in range(n):
        position = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ])
        velocity = np.zeros(2)
        ped_type = np.random.choice(types, p=probs)

        ped = Pedestrian.create_with_type(
            id=i,
            position=position,
            velocity=velocity,
            target=target.copy(),
            ped_type=ped_type,
            speed_variation=True
        )
        pedestrians.append(ped)

    return pedestrians


def create_pedestrian_with_type(
    id: int,
    position: np.ndarray,
    target: np.ndarray,
    ped_type: PedestrianType = PedestrianType.NORMAL
) -> Pedestrian:
    """便捷函数：创建指定类型的单个行人"""
    return Pedestrian.create_with_type(
        id=id,
        position=position,
        velocity=np.zeros(2),
        target=target,
        ped_type=ped_type,
        speed_variation=True
    )
