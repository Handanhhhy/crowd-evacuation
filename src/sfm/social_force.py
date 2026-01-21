"""
社会力模型实现
基于 Helbing 的社会力模型，结合 pysocialforce 库

文献参考:
- Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
- Weidmann, U. (1993). Transporttechnik der Fußgänger.
- Fruin, J. J. (1971). Pedestrian planning and design.
- Hall, E. T. (1966). The Hidden Dimension.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class PedestrianType(Enum):
    """行人类型枚举

    基于文献的行人类型分类:
    - NORMAL: 普通成年人 (Helbing 1995)
    - ELDERLY: 老年人 (Weidmann 1993)
    - CHILD: 儿童 (Fruin 1971)
    - IMPATIENT: 急躁型 (参考 Helbing)
    """
    NORMAL = "normal"      # 普通成年人
    ELDERLY = "elderly"    # 老人
    CHILD = "child"        # 儿童
    IMPATIENT = "impatient" # 急躁型


# 行人类型参数配置 (基于文献)
# 参考: Helbing 1995, Weidmann 1993, Fruin 1971
PEDESTRIAN_TYPE_PARAMS = {
    PedestrianType.NORMAL: {
        'desired_speed': 1.34,      # Helbing 1995: 1.34 m/s
        'speed_std': 0.26,          # Helbing 1995: σ = 0.26
        'reaction_time': 0.5,       # Helbing 2000: τ = 0.5s
        'radius': 0.3,
        'color': 'blue',            # 可视化颜色
    },
    PedestrianType.ELDERLY: {
        'desired_speed': 0.9,       # Weidmann 1993: 0.8-1.0 m/s
        'speed_std': 0.15,
        'reaction_time': 0.8,       # 反应时间较长
        'radius': 0.3,
        'color': 'green',
    },
    PedestrianType.CHILD: {
        'desired_speed': 0.7,       # Fruin 1971: 0.6-0.8 m/s
        'speed_std': 0.2,
        'reaction_time': 0.6,
        'radius': 0.25,             # 儿童体型较小
        'color': 'yellow',
    },
    PedestrianType.IMPATIENT: {
        'desired_speed': 1.6,       # 参考 Helbing, 较高速度
        'speed_std': 0.2,
        'reaction_time': 0.3,       # 反应更快
        'radius': 0.3,
        'color': 'red',
    },
}


@dataclass
class Pedestrian:
    """行人状态

    扩展版本，支持行人类型和增强行为特征
    """
    id: int
    position: np.ndarray      # [x, y]
    velocity: np.ndarray      # [vx, vy]
    target: np.ndarray        # 目标位置 [x, y]
    desired_speed: float = 1.34  # 期望速度 m/s
    radius: float = 0.3       # 行人半径 m
    ped_type: PedestrianType = PedestrianType.NORMAL  # 行人类型
    reaction_time: float = 0.5  # 反应时间 (tau)

    # 行为状态
    is_waiting: bool = False  # 是否在等待
    wait_timer: float = 0.0   # 等待计时器
    panic_factor: float = 0.0  # 恐慌因子 (0-1)
    stuck_timer: float = 0.0  # 卡住计时器（用于反堵塞）
    last_position: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 上一位置

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
        """根据类型创建行人，自动应用文献参数

        Args:
            id: 行人ID
            position: 初始位置
            velocity: 初始速度
            target: 目标位置
            ped_type: 行人类型
            speed_variation: 是否添加速度随机变化
        """
        params = PEDESTRIAN_TYPE_PARAMS[ped_type]

        # 期望速度 (添加随机变化)
        if speed_variation:
            desired_speed = np.random.normal(
                params['desired_speed'],
                params['speed_std']
            )
            # 确保速度在合理范围内
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
            last_position=position.copy()
        )


class SocialForceModel:
    """增强版社会力模型

    F_total = F_drive + F_social + F_obstacle + F_learned

    - F_drive: 驱动力，驱使行人向目标移动
    - F_social: 社会力，行人之间的排斥力
    - F_obstacle: 障碍物排斥力
    - F_learned: GBM学习的行为修正（可选）

    增强特性:
    - 行人类型差异化（老人、儿童、急躁型）
    - 等待/排队行为
    - 恐慌反应
    - GBM行为预测（基于ETH/UCY真实数据训练）

    文献参考:
    - Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
    - Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features of escape panic.
    - ETH/UCY数据集: 真实行人轨迹数据，用于行为学习
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
        # 增强行为参数
        enable_waiting: bool = True,           # 启用等待行为
        waiting_density_threshold: float = 0.8, # 触发等待的密度阈值
        enable_perturbation: bool = True,       # 启用随机扰动
        perturbation_sigma: float = 0.1,        # 随机扰动标准差
        enable_panic: bool = True,              # 启用恐慌反应
        panic_density_threshold: float = 1.5,   # 触发恐慌的密度阈值
        # GBM行为预测器
        gbm_predictor = None,      # GBM模型，用于学习行为
        gbm_weight: float = 0.3,   # GBM预测权重 (0-1)
    ):
        self.tau = tau
        self.A = A
        self.B = B
        self.k = k
        self.kappa = kappa
        self.wall_A = wall_A
        self.wall_B = wall_B

        # 增强行为参数
        self.enable_waiting = enable_waiting
        self.waiting_density_threshold = waiting_density_threshold
        self.enable_perturbation = enable_perturbation
        self.perturbation_sigma = perturbation_sigma
        self.enable_panic = enable_panic
        self.panic_density_threshold = panic_density_threshold

        # GBM行为预测器（基于ETH/UCY真实数据训练）
        self.gbm_predictor = gbm_predictor
        self.gbm_weight = gbm_weight  # 融合权重: 0=纯SFM, 1=纯GBM

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

        增强特性:
        - 使用行人类型特定的反应时间 (tau)
        - 支持等待行为 (速度降为0)
        - 支持恐慌因子 (速度增加)
        """
        # 如果行人在等待，驱动力为减速力
        if ped.is_waiting:
            # 驱使行人停下来
            return -ped.velocity / ped.reaction_time

        direction = ped.target - ped.position
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # 到达目标
            return np.zeros(2)

        e = direction / distance  # 单位方向向量

        # 计算有效期望速度 (考虑恐慌因子)
        # 恐慌时速度增加: v = v0 × (1 + panic_factor)
        effective_speed = ped.desired_speed * (1.0 + ped.panic_factor)

        # 限制最大速度 (即使恐慌也不能超过2.5 m/s)
        effective_speed = min(effective_speed, 2.5)

        desired_velocity = effective_speed * e

        # 使用行人类型特定的反应时间
        tau = ped.reaction_time

        return (desired_velocity - ped.velocity) / tau

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
        """计算障碍物排斥力（增强版）"""
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
                # 如果几乎在障碍物上，给一个随机方向的强推力
                random_dir = np.random.uniform(-1, 1, 2)
                random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-6)
                force += random_dir * 5000
                continue

            n = diff / distance

            # 排斥力（距离越近力越大）
            force += self.wall_A * np.exp((ped.radius - distance) / self.wall_B) * n

            # 接触力 - 当行人接触或穿透障碍物时
            if distance < ped.radius:
                force += self.k * (ped.radius - distance) * n

            # 额外的近距离强推力 - 防止卡在障碍物边缘
            if distance < ped.radius * 1.5:
                # 增加额外的推力，距离越近越强
                extra_force = 3000 * (1 - distance / (ped.radius * 1.5))
                force += extra_force * n

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

    def compute_local_density(self, ped: Pedestrian, radius: float = 2.0) -> float:
        """计算行人周围的局部密度

        Args:
            ped: 当前行人
            radius: 检测半径（默认2米，符合Hall 1966社会距离理论）

        Returns:
            密度值（人/平方米）
        """
        count = 0
        for other in self.pedestrians:
            if other.id == ped.id:
                continue
            distance = np.linalg.norm(ped.position - other.position)
            if distance < radius:
                count += 1

        # 密度 = 人数 / 面积
        area = np.pi * radius ** 2
        return count / area

    def compute_random_perturbation(self, ped: Pedestrian) -> np.ndarray:
        """计算随机扰动力（模拟犹豫行为）

        基于高斯噪声，σ = 0.1 m/s（可配置）
        参考: Helbing 2000 - 行人行为的随机性
        """
        if not self.enable_perturbation:
            return np.zeros(2)

        # 高斯随机扰动
        perturbation = np.random.normal(0, self.perturbation_sigma, 2)

        return perturbation

    def compute_gbm_velocity_correction(self, ped: Pedestrian) -> np.ndarray:
        """计算GBM预测的速度修正

        使用训练好的GBM模型（基于ETH/UCY真实轨迹数据）
        预测更真实的行人运动。

        GBM模型训练特征:
        - 位置 (pos_x, pos_y)
        - 速度 (vel_x, vel_y, speed_mean, speed_std, speed_last)
        - 加速度 (acc_x, acc_y)
        - 方向 (direction, direction_change_mean)
        - 轨迹形状 (displacement, path_length)

        Returns:
            速度修正向量 [dvx, dvy]
        """
        if self.gbm_predictor is None:
            return np.zeros(2)

        try:
            # 准备GBM预测特征
            speed = np.linalg.norm(ped.velocity)
            direction = np.arctan2(ped.velocity[1], ped.velocity[0]) if speed > 0.01 else 0

            # 计算位移和路径指标
            direction_to_target = ped.target - ped.position
            displacement = np.linalg.norm(direction_to_target)

            # 构建特征向量（必须与训练特征匹配）
            features = np.array([
                ped.position[0],      # pos_x
                ped.position[1],      # pos_y
                ped.velocity[0],      # vel_x
                ped.velocity[1],      # vel_y
                speed,                # speed_mean（用当前值近似）
                0.1,                  # speed_std（默认值）
                speed,                # speed_last
                0.0,                  # acc_x（近似）
                0.0,                  # acc_y（近似）
                direction,            # direction
                0.0,                  # direction_change_mean
                displacement,         # displacement
                speed * 0.1,          # path_length（近似）
            ]).reshape(1, -1)

            # GBM预测: [target_x, target_y, target_vx, target_vy]
            prediction = self.gbm_predictor.predict(features)[0]

            # 提取速度修正（后2个值: target_vx, target_vy）
            velocity_correction = np.array([prediction[2], prediction[3]])

            # 限制修正幅度，避免不稳定
            correction_magnitude = np.linalg.norm(velocity_correction)
            max_correction = 0.5  # 最大修正 0.5 m/s
            if correction_magnitude > max_correction:
                velocity_correction = velocity_correction / correction_magnitude * max_correction

            return velocity_correction

        except Exception as e:
            # 预测失败时返回零修正
            return np.zeros(2)

    def update_waiting_state(self, ped: Pedestrian, dt: float) -> None:
        """更新行人的等待状态

        当前方密度超过阈值时，行人进入等待状态

        Args:
            ped: 行人对象
            dt: 时间步长
        """
        if not self.enable_waiting:
            ped.is_waiting = False
            return

        # 计算行进方向前方的密度
        direction = ped.target - ped.position
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            ped.is_waiting = False
            return

        e = direction / dist

        # 检测前方扇形区域内的行人数量
        front_count = 0
        for other in self.pedestrians:
            if other.id == ped.id:
                continue
            diff = other.position - ped.position
            distance = np.linalg.norm(diff)

            # 只检测前方3米范围内
            if distance < 3.0 and distance > 0.1:
                # 检查是否在前方 (夹角小于60度)
                cos_angle = np.dot(diff, e) / distance
                if cos_angle > 0.5:  # cos(60°) = 0.5
                    front_count += 1

        # 计算前方密度
        front_density = front_count / (np.pi * 3.0 ** 2 / 2)  # 半圆面积

        # 判断是否需要等待
        if front_density > self.waiting_density_threshold:
            ped.is_waiting = True
            ped.wait_timer += dt
        else:
            ped.is_waiting = False
            ped.wait_timer = 0.0

        # 等待时间过长自动恢复 (最多等待1.5秒，减少死锁)
        if ped.wait_timer > 1.5:
            ped.is_waiting = False
            ped.wait_timer = 0.0

        # 如果行人卡住太久，强制取消等待状态
        if hasattr(ped, 'stuck_timer') and ped.stuck_timer > 0.5:
            ped.is_waiting = False

    def update_panic_factor(self, ped: Pedestrian) -> None:
        """更新行人的恐慌因子

        当周围密度过高时，恐慌因子增加
        恐慌因子影响期望速度: v = v0 × (1 + panic_factor)

        参考: Helbing 2000 - Escape panic dynamics
        """
        if not self.enable_panic:
            ped.panic_factor = 0.0
            return

        # 计算局部密度
        density = self.compute_local_density(ped, radius=2.0)

        # 恐慌因子与密度相关
        if density > self.panic_density_threshold:
            # 线性增加恐慌因子，最大0.5
            ped.panic_factor = min(
                (density - self.panic_density_threshold) * 0.3,
                0.5
            )
        else:
            # 逐渐恢复平静
            ped.panic_factor = max(0, ped.panic_factor - 0.02)

    def compute_total_force(self, ped: Pedestrian) -> np.ndarray:
        """计算总力

        F_total = F_drive + F_social + F_obstacle + F_random
        """
        f_drive = self.compute_driving_force(ped)
        f_social = self.compute_social_force(ped)
        f_obstacle = self.compute_obstacle_force(ped)
        f_random = self.compute_random_perturbation(ped)

        return f_drive + f_social + f_obstacle + f_random

    def step(self, dt: float = 0.1) -> None:
        """更新所有行人状态（一个时间步）

        集成GBM的增强版本:
        1. 更新行为状态（等待、恐慌）
        2. 计算SFM社会力
        3. 应用GBM速度修正（从真实数据学习）
        4. 更新速度和位置

        GBM修正替代了之前的随机反卡住机制，
        提供从ETH/UCY数据学习的更真实的避障行为。
        """
        # 1. 更新行为状态
        for ped in self.pedestrians:
            self.update_waiting_state(ped, dt)
            self.update_panic_factor(ped)

        # 2. 计算所有行人的SFM力
        forces = [self.compute_total_force(ped) for ped in self.pedestrians]

        # 3. 计算GBM速度修正（如果预测器可用）
        gbm_corrections = []
        if self.gbm_predictor is not None:
            gbm_corrections = [self.compute_gbm_velocity_correction(ped)
                              for ped in self.pedestrians]
        else:
            gbm_corrections = [np.zeros(2) for _ in self.pedestrians]

        # 4. 更新速度和位置
        for ped, force, gbm_corr in zip(self.pedestrians, forces, gbm_corrections):
            # 计算SFM速度更新
            sfm_velocity = ped.velocity + force * dt

            # 融合SFM与GBM预测
            # gbm_weight=0: 纯SFM, gbm_weight=1: 纯GBM
            if self.gbm_predictor is not None and np.linalg.norm(gbm_corr) > 0.01:
                # GBM提供速度修正，使运动更自然
                # 在行人靠近障碍物时特别有用
                blended_velocity = (1 - self.gbm_weight) * sfm_velocity + \
                                   self.gbm_weight * (ped.velocity + gbm_corr)
                ped.velocity = blended_velocity
            else:
                ped.velocity = sfm_velocity

            # 限制最大速度（考虑恐慌因子）
            speed = np.linalg.norm(ped.velocity)
            max_speed = ped.desired_speed * (1.5 + ped.panic_factor * 0.3)
            if speed > max_speed:
                ped.velocity = ped.velocity / speed * max_speed

            # ========== 增强版反堵塞机制 ==========
            # 基于实际位移检测卡住，而不仅是速度
            actual_movement = np.linalg.norm(ped.position - ped.last_position)
            direction = ped.target - ped.position
            dist_to_target = np.linalg.norm(direction)

            # 检测是否卡住：移动很少且不是在等待且还没到达目标
            is_stuck = actual_movement < 0.05 * dt and not ped.is_waiting and dist_to_target > 1.0

            if is_stuck:
                ped.stuck_timer += dt
            else:
                # 快速衰减卡住计时器
                ped.stuck_timer = max(0, ped.stuck_timer - dt * 2)

            # 根据卡住时间采取不同强度的脱困措施
            if ped.stuck_timer > 0.2 and dist_to_target > 0.5:
                # 计算脱困强度（卡住越久，力度越大）
                stuck_level = min(ped.stuck_timer / 1.5, 1.0)  # 0-1，1.5秒达到最大

                # 方案1：轻度卡住 (0.2-0.8秒) - 随机扰动
                if ped.stuck_timer < 0.8:
                    perturbation = np.random.uniform(-0.8, 0.8, 2)
                    ped.velocity += perturbation * (1 + stuck_level)
                    # 增加向目标的推力
                    if dist_to_target > 0.1:
                        ped.velocity += 0.5 * direction / dist_to_target

                # 方案2：中度卡住 (0.8-1.5秒) - 横向逃逸
                elif ped.stuck_timer < 1.5:
                    if dist_to_target > 0.1:
                        # 计算垂直于目标方向的逃逸方向
                        escape_dir = np.array([-direction[1], direction[0]]) / dist_to_target
                        # 随机选择左或右
                        if np.random.random() > 0.5:
                            escape_dir = -escape_dir
                        ped.velocity += escape_dir * 1.2
                        ped.velocity += 0.6 * direction / dist_to_target

                # 方案3：严重卡住 (>1.5秒) - 强制推开 + 逃离人群
                else:
                    # 强力随机推动
                    strong_push = np.random.uniform(-1.5, 1.5, 2)
                    ped.velocity += strong_push

                    # 向密度较低方向移动
                    if dist_to_target > 0.1:
                        # 检测周围行人，找到人少的方向
                        away_from_crowd = np.zeros(2)
                        for other in self.pedestrians:
                            if other.id != ped.id:
                                diff = ped.position - other.position
                                dist_other = np.linalg.norm(diff)
                                if 0.1 < dist_other < 3.0:
                                    away_from_crowd += diff / (dist_other ** 2)

                        if np.linalg.norm(away_from_crowd) > 0.1:
                            away_from_crowd = away_from_crowd / np.linalg.norm(away_from_crowd)
                            ped.velocity += away_from_crowd * 1.0

                        # 仍然保持向目标的倾向
                        ped.velocity += 0.8 * direction / dist_to_target

            # 保存当前位置用于下次比较
            ped.last_position = ped.position.copy()

            # 更新位置
            new_position = ped.position + ped.velocity * dt

            # 障碍物碰撞检测和修正
            # 检查新位置是否会导致与障碍物碰撞
            collision_resolved = False
            for obstacle in self.obstacles:
                start, end = obstacle[0], obstacle[1]
                closest_point = self._closest_point_on_segment(new_position, start, end)
                dist_to_obstacle = np.linalg.norm(new_position - closest_point)

                if dist_to_obstacle < ped.radius * 0.8:
                    # 碰撞！将行人推离障碍物
                    if dist_to_obstacle > 1e-6:
                        push_dir = (new_position - closest_point) / dist_to_obstacle
                    else:
                        push_dir = np.random.uniform(-1, 1, 2)
                        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)

                    # 将行人推到安全距离
                    new_position = closest_point + push_dir * (ped.radius * 1.2)
                    # 调整速度方向远离障碍物
                    ped.velocity = ped.velocity - 2 * np.dot(ped.velocity, -push_dir) * (-push_dir)
                    collision_resolved = True

            ped.position = new_position

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
    seed: int = None,
    type_distribution: Optional[dict] = None
) -> List[Pedestrian]:
    """创建随机分布的行人

    Args:
        n: 行人数量
        spawn_area: 生成区域 (x_min, y_min, x_max, y_max)
        target: 目标位置
        seed: 随机种子
        type_distribution: 行人类型分布比例，例如:
            {
                PedestrianType.NORMAL: 0.70,
                PedestrianType.ELDERLY: 0.15,
                PedestrianType.CHILD: 0.10,
                PedestrianType.IMPATIENT: 0.05
            }
    """
    if seed is not None:
        np.random.seed(seed)

    # 默认类型分布 (基于一般人群构成)
    if type_distribution is None:
        type_distribution = {
            PedestrianType.NORMAL: 0.70,
            PedestrianType.ELDERLY: 0.15,
            PedestrianType.CHILD: 0.10,
            PedestrianType.IMPATIENT: 0.05,
        }

    # 根据分布生成类型列表
    types = list(type_distribution.keys())
    probs = list(type_distribution.values())
    # 归一化概率
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

        # 随机选择行人类型
        ped_type = np.random.choice(types, p=probs)

        # 使用工厂方法创建带类型的行人
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
    """便捷函数：创建指定类型的单个行人

    Args:
        id: 行人ID
        position: 初始位置
        target: 目标位置
        ped_type: 行人类型
    """
    return Pedestrian.create_with_type(
        id=id,
        position=position,
        velocity=np.zeros(2),
        target=target,
        ped_type=ped_type,
        speed_variation=True
    )
