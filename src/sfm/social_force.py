"""
社会力模型实现
基于 Helbing 的社会力模型，结合 pysocialforce 库
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Pedestrian:
    """行人状态"""
    id: int
    position: np.ndarray      # [x, y]
    velocity: np.ndarray      # [vx, vy]
    target: np.ndarray        # 目标位置 [x, y]
    desired_speed: float = 1.34  # 期望速度 m/s
    radius: float = 0.3       # 行人半径 m

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)


class SocialForceModel:
    """社会力模型

    F_total = F_drive + F_social + F_obstacle

    - F_drive: 驱动力，驱使行人向目标移动
    - F_social: 社会力，行人之间的排斥力
    - F_obstacle: 障碍物排斥力
    """

    def __init__(
        self,
        tau: float = 0.5,          # 松弛时间
        A: float = 2000.0,         # 社会力强度
        B: float = 0.08,           # 社会力范围
        k: float = 1.2e5,          # 身体力常数
        kappa: float = 2.4e5,      # 摩擦力常数
        wall_A: float = 2000.0,    # 墙壁排斥力强度
        wall_B: float = 0.08,      # 墙壁排斥力范围
    ):
        self.tau = tau
        self.A = A
        self.B = B
        self.k = k
        self.kappa = kappa
        self.wall_A = wall_A
        self.wall_B = wall_B

        self.pedestrians: List[Pedestrian] = []
        self.obstacles: List[np.ndarray] = []  # 障碍物线段列表

    def add_pedestrian(self, ped: Pedestrian) -> None:
        """添加行人"""
        self.pedestrians.append(ped)

    def add_obstacle(self, start: np.ndarray, end: np.ndarray) -> None:
        """添加障碍物（线段）"""
        self.obstacles.append(np.array([start, end]))

    def compute_driving_force(self, ped: Pedestrian) -> np.ndarray:
        """计算驱动力 F_drive = (v0 * e - v) / tau

        驱使行人以期望速度向目标方向移动
        """
        direction = ped.target - ped.position
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # 到达目标
            return np.zeros(2)

        e = direction / distance  # 单位方向向量
        desired_velocity = ped.desired_speed * e

        return (desired_velocity - ped.velocity) / self.tau

    def compute_social_force(self, ped: Pedestrian) -> np.ndarray:
        """计算社会力（行人之间的排斥力）

        F_social = A * exp((r_ij - d_ij) / B) * n_ij
        """
        force = np.zeros(2)

        for other in self.pedestrians:
            if other.id == ped.id:
                continue

            # 计算距离
            diff = ped.position - other.position
            distance = np.linalg.norm(diff)

            if distance < 1e-6:
                continue

            n_ij = diff / distance  # 单位向量
            r_ij = ped.radius + other.radius  # 半径和

            # 排斥力
            force += self.A * np.exp((r_ij - distance) / self.B) * n_ij

            # 如果发生接触，添加身体力和摩擦力
            if distance < r_ij:
                # 身体力
                force += self.k * (r_ij - distance) * n_ij

                # 摩擦力（切向）
                t_ij = np.array([-n_ij[1], n_ij[0]])  # 切向向量
                delta_v = np.dot(other.velocity - ped.velocity, t_ij)
                force += self.kappa * (r_ij - distance) * delta_v * t_ij

        return force

    def compute_obstacle_force(self, ped: Pedestrian) -> np.ndarray:
        """计算障碍物排斥力"""
        force = np.zeros(2)

        for obstacle in self.obstacles:
            # 计算点到线段的最近距离
            start, end = obstacle[0], obstacle[1]
            closest_point = self._closest_point_on_segment(
                ped.position, start, end
            )

            diff = ped.position - closest_point
            distance = np.linalg.norm(diff)

            if distance < 1e-6:
                continue

            n = diff / distance

            # 排斥力
            force += self.wall_A * np.exp((ped.radius - distance) / self.wall_B) * n

            # 接触力
            if distance < ped.radius:
                force += self.k * (ped.radius - distance) * n

        return force

    def _closest_point_on_segment(
        self,
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray
    ) -> np.ndarray:
        """计算点到线段的最近点"""
        segment = end - start
        length_sq = np.dot(segment, segment)

        if length_sq < 1e-6:
            return start

        t = max(0, min(1, np.dot(point - start, segment) / length_sq))
        return start + t * segment

    def compute_total_force(self, ped: Pedestrian) -> np.ndarray:
        """计算总力"""
        f_drive = self.compute_driving_force(ped)
        f_social = self.compute_social_force(ped)
        f_obstacle = self.compute_obstacle_force(ped)

        return f_drive + f_social + f_obstacle

    def step(self, dt: float = 0.1) -> None:
        """更新所有行人状态（一个时间步）"""
        # 计算所有行人的力
        forces = [self.compute_total_force(ped) for ped in self.pedestrians]

        # 更新速度和位置
        for ped, force in zip(self.pedestrians, forces):
            # 更新速度 (F = ma, 假设质量为1)
            ped.velocity = ped.velocity + force * dt

            # 限制最大速度
            speed = np.linalg.norm(ped.velocity)
            max_speed = ped.desired_speed * 1.5
            if speed > max_speed:
                ped.velocity = ped.velocity / speed * max_speed

            # 更新位置
            ped.position = ped.position + ped.velocity * dt

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
        return np.array(states)

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
    seed: int = None
) -> List[Pedestrian]:
    """创建随机分布的行人

    Args:
        n: 行人数量
        spawn_area: 生成区域 (x_min, y_min, x_max, y_max)
        target: 目标位置
        seed: 随机种子
    """
    if seed is not None:
        np.random.seed(seed)

    pedestrians = []
    x_min, y_min, x_max, y_max = spawn_area

    for i in range(n):
        position = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ])
        velocity = np.zeros(2)

        ped = Pedestrian(
            id=i,
            position=position,
            velocity=velocity,
            target=target.copy(),
            desired_speed=np.random.uniform(1.0, 1.5)  # 速度略有差异
        )
        pedestrians.append(ped)

    return pedestrians
