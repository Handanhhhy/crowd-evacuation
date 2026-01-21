"""
成都东客站地铁出站口强化学习环境
适配3出口、闸机、柱子的复杂场景

增强版本:
- 支持多种行人类型 (老人、儿童、急躁型等)
- 可选GBM行为预测修正
- 更真实的行人行为 (等待、犹豫、恐慌)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import joblib

# 导入社会力模型
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sfm.social_force import (
    SocialForceModel,
    Pedestrian,
    PedestrianType,
    PEDESTRIAN_TYPE_PARAMS
)


@dataclass
class Exit:
    """出口定义"""
    id: int
    name: str
    position: np.ndarray
    width: float


class MetroEvacuationEnv(gym.Env):
    """成都东客站地铁出站口疏散环境

    观测空间 (8维):
        - 3个出口的密度 (3)
        - 3个出口的拥堵度 (3)
        - 剩余人数比例 (1)
        - 时间比例 (1)

    动作空间: Discrete(3) - 选择推荐出口A/B/C

    奖励函数:
        - 疏散奖励: +10 × 新疏散人数
        - 拥堵惩罚: -2 × 总拥堵度
        - 时间惩罚: -0.1/步
        - 完成奖励: +100
        - 均衡奖励: 鼓励各出口分流
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        n_pedestrians: int = 80,
        scene_size: Tuple[float, float] = (60.0, 40.0),
        max_steps: int = 1000,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
        # 增强行人行为参数
        type_distribution: Optional[Dict[PedestrianType, float]] = None,
        enable_enhanced_behaviors: bool = True,
        gbm_model_path: Optional[str] = None,
    ):
        """
        Args:
            n_pedestrians: 行人数量
            scene_size: 场景尺寸 (宽, 高)
            max_steps: 最大仿真步数
            dt: 时间步长
            render_mode: 渲染模式
            type_distribution: 行人类型分布比例
                默认: 70%普通 + 15%老人 + 10%儿童 + 5%急躁
            enable_enhanced_behaviors: 是否启用增强行为 (等待、犹豫、恐慌)
            gbm_model_path: GBM行为预测模型路径 (可选)
        """
        super().__init__()

        self.n_pedestrians = n_pedestrians
        self.scene_width, self.scene_height = scene_size
        self.max_steps = max_steps
        self.dt = dt
        self.render_mode = render_mode

        # 增强行为配置
        self.enable_enhanced_behaviors = enable_enhanced_behaviors

        # 行人类型分布 (默认比例基于一般人群构成)
        self.type_distribution = type_distribution or {
            PedestrianType.NORMAL: 0.70,
            PedestrianType.ELDERLY: 0.15,
            PedestrianType.CHILD: 0.10,
            PedestrianType.IMPATIENT: 0.05,
        }

        # GBM行为预测器 (可选)
        self.gbm_predictor = None
        self.gbm_model_path = gbm_model_path
        if gbm_model_path:
            self._load_gbm_predictor(gbm_model_path)

        # 定义3个出口
        self.exits = [
            Exit(id=0, name='A', position=np.array([60.0, 10.0]), width=4.0),
            Exit(id=1, name='B', position=np.array([60.0, 30.0]), width=4.0),
            Exit(id=2, name='C', position=np.array([40.0, 40.0]), width=5.0),
        ]
        self.n_exits = len(self.exits)

        # 闸机位置
        self.gate_y_positions = [10, 15, 20, 25, 30]

        # 柱子位置
        self.pillars = [
            np.array([30.0, 12.0]), np.array([30.0, 28.0]),
            np.array([45.0, 12.0]), np.array([45.0, 28.0]),
            np.array([35.0, 20.0]), np.array([50.0, 20.0]),
        ]

        # 设施
        self.facilities = [
            {'position': np.array([32.0, 20.0]), 'size': (3.0, 2.0)},
            {'position': np.array([55.0, 20.0]), 'size': (4.0, 8.0)},
        ]

        # 观测空间: [3个出口密度, 3个出口拥堵度, 剩余人数比例, 时间比例]
        obs_dim = self.n_exits * 2 + 2  # 3*2 + 2 = 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间: 选择推荐出口 (A/B/C)
        self.action_space = spaces.Discrete(self.n_exits)

        # 初始化
        self.sfm = None
        self.current_step = 0
        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.evacuated_by_type = {t: 0 for t in PedestrianType}  # 按类型统计

        # 记录
        self.history = {
            'evacuated': [],
            'congestion': [],
            'rewards': []
        }

    def _load_gbm_predictor(self, model_path: str) -> bool:
        """加载GBM行为预测模型

        Args:
            model_path: 模型文件路径

        Returns:
            是否加载成功
        """
        try:
            from ml.gbm_predictor import GBMPredictor
            self.gbm_predictor = GBMPredictor()
            self.gbm_predictor.load(model_path)
            print(f"GBM行为预测模型已加载: {model_path}")
            return True
        except Exception as e:
            print(f"GBM模型加载失败: {e}")
            self.gbm_predictor = None
            return False

    def _create_sfm(self) -> SocialForceModel:
        """创建社会力模型实例（包含地铁站场景的障碍物）

        如果启用增强行为，将开启等待、犹豫、恐慌等特性
        """
        sfm = SocialForceModel(
            tau=0.5,
            A=2000.0,
            B=0.08,
            wall_A=5000.0,    # Increased to prevent pedestrians getting stuck
            wall_B=0.1,       # Increased range for earlier obstacle detection
            # Enhanced behavior parameters
            enable_waiting=self.enable_enhanced_behaviors,
            enable_perturbation=self.enable_enhanced_behaviors,
            enable_panic=self.enable_enhanced_behaviors,
            waiting_density_threshold=0.8,
            perturbation_sigma=0.05,  # Reduced since GBM handles behavior
            panic_density_threshold=1.5,
            # GBM behavior predictor (trained on ETH/UCY real data)
            gbm_predictor=self.gbm_predictor,
            gbm_weight=0.3,   # 30% GBM, 70% SFM blend
        )

        # 外墙
        # 上墙（有出口C的间隔）
        sfm.add_obstacle(np.array([0, self.scene_height]), np.array([35, self.scene_height]))
        sfm.add_obstacle(np.array([45, self.scene_height]), np.array([self.scene_width, self.scene_height]))

        # 下墙
        sfm.add_obstacle(np.array([0, 0]), np.array([self.scene_width, 0]))

        # 左墙
        sfm.add_obstacle(np.array([0, 0]), np.array([0, self.scene_height]))

        # 右墙（有出口A、B的间隔）
        sfm.add_obstacle(np.array([self.scene_width, 0]), np.array([self.scene_width, 8]))
        sfm.add_obstacle(np.array([self.scene_width, 12]), np.array([self.scene_width, 28]))
        sfm.add_obstacle(np.array([self.scene_width, 32]), np.array([self.scene_width, self.scene_height]))

        # 闸机隔板
        barrier_y_positions = [7, 12.5, 17.5, 22.5, 27.5, 33]
        for by in barrier_y_positions:
            sfm.add_obstacle(np.array([18.5, by]), np.array([21.5, by]))

        # 柱子
        for pos in self.pillars:
            size = 0.8
            sfm.add_obstacle(pos - np.array([size, 0]), pos + np.array([size, 0]))
            sfm.add_obstacle(pos - np.array([0, size]), pos + np.array([0, size]))

        # 设施障碍
        for facility in self.facilities:
            pos = facility['position']
            w, h = facility['size']
            sfm.add_obstacle(np.array([pos[0] - w/2, pos[1] - h/2]), np.array([pos[0] + w/2, pos[1] - h/2]))
            sfm.add_obstacle(np.array([pos[0] - w/2, pos[1] + h/2]), np.array([pos[0] + w/2, pos[1] + h/2]))
            sfm.add_obstacle(np.array([pos[0] - w/2, pos[1] - h/2]), np.array([pos[0] - w/2, pos[1] + h/2]))
            sfm.add_obstacle(np.array([pos[0] + w/2, pos[1] - h/2]), np.array([pos[0] + w/2, pos[1] + h/2]))

        return sfm

    def _spawn_pedestrians(self, target_exit_id: int = 2):
        """生成行人（从站台区域）

        使用增强版本：
        - 根据配置的类型分布创建不同类型的行人
        - 老人、儿童、急躁型等具有不同的速度和行为特征

        文献参数:
        - NORMAL: 1.34 m/s (Helbing 1995)
        - ELDERLY: 0.9 m/s (Weidmann 1993)
        - CHILD: 0.7 m/s (Fruin 1971)
        - IMPATIENT: 1.6 m/s
        """
        target = self.exits[target_exit_id].position

        # 准备类型分布
        types = list(self.type_distribution.keys())
        probs = list(self.type_distribution.values())
        # 归一化概率
        total = sum(probs)
        probs = [p / total for p in probs]

        for i in range(self.n_pedestrians):
            # 在站台区域随机生成
            position = np.array([
                np.random.uniform(2, 14),
                np.random.uniform(12, 28)
            ])
            velocity = np.zeros(2)

            # 初始目标根据位置智能选择
            y = position[1]
            if y < 15:
                weights = [0.5, 0.2, 0.3]  # 偏向出口A
            elif y > 25:
                weights = [0.2, 0.5, 0.3]  # 偏向出口B
            else:
                weights = [0.2, 0.2, 0.6]  # 偏向主出口C

            choice = np.random.choice([0, 1, 2], p=weights)
            initial_target = self.exits[choice].position.copy()

            # 随机选择行人类型
            ped_type = np.random.choice(types, p=probs)

            # 使用工厂方法创建带类型的行人
            ped = Pedestrian.create_with_type(
                id=i,
                position=position,
                velocity=velocity,
                target=initial_target,
                ped_type=ped_type,
                speed_variation=True
            )
            self.sfm.add_pedestrian(ped)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 重新创建社会力模型
        self.sfm = self._create_sfm()

        # 生成行人
        self._spawn_pedestrians()

        self.current_step = 0
        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.evacuated_by_type = {t: 0 for t in PedestrianType}

        self.history = {
            'evacuated': [],
            'congestion': [],
            'rewards': []
        }

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """获取观测 (8维)"""
        exit_densities = []
        exit_congestions = []

        for exit_obj in self.exits:
            density, congestion = self._compute_exit_metrics(exit_obj)
            exit_densities.append(density)
            exit_congestions.append(congestion)

        # 剩余人数比例
        remaining_ratio = len(self.sfm.pedestrians) / max(self.n_pedestrians, 1)

        # 时间比例
        time_ratio = self.current_step / self.max_steps

        obs = np.array(
            exit_densities + exit_congestions + [remaining_ratio, time_ratio],
            dtype=np.float32
        )

        return np.clip(obs, 0.0, 1.0)

    def _compute_exit_metrics(self, exit_obj: Exit) -> Tuple[float, float]:
        """计算出口附近的密度和拥堵度"""
        radius = 8.0  # 检测半径（地铁站场景更大）
        exit_pos = exit_obj.position

        nearby_peds = []
        for ped in self.sfm.pedestrians:
            dist = np.linalg.norm(ped.position - exit_pos)
            if dist < radius:
                nearby_peds.append(ped)

        # 密度 (归一化到 0-1)
        density = min(len(nearby_peds) / 25.0, 1.0)

        # 拥堵度 (基于平均速度下降)
        if len(nearby_peds) > 0:
            avg_speed = np.mean([ped.speed for ped in nearby_peds])
            expected_speed = 1.2
            congestion = max(0, 1 - avg_speed / expected_speed)
        else:
            congestion = 0.0

        return density, congestion

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步

        Args:
            action: 选择的主要引导出口 ID (0=A, 1=B, 2=C)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 根据动作调整部分行人的目标出口
        self._apply_action(action)

        # 运行社会力模型多步
        for _ in range(5):
            self.sfm.step(self.dt)

        # 检查并移除已到达出口的行人
        self._check_evacuated()

        self.current_step += 1

        # 计算奖励
        reward = self._compute_reward()
        self.history['rewards'].append(reward)

        # 检查终止条件
        terminated = len(self.sfm.pedestrians) == 0
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            'evacuated': self.evacuated_count,
            'remaining': len(self.sfm.pedestrians),
            'step': self.current_step,
            'evacuated_by_exit': self.evacuated_by_exit.copy(),
            'evacuated_by_type': {t.value: c for t, c in self.evacuated_by_type.items()},
            'enhanced_behaviors': self.enable_enhanced_behaviors,
        }

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int):
        """应用动作：引导行人到推荐出口"""
        target_exit = self.exits[action]

        for ped in self.sfm.pedestrians:
            # 只有在大厅区域的行人才响应引导（已过闸机）
            if ped.position[0] > 22:
                # 计算到当前目标和推荐目标的距离
                dist_to_current = np.linalg.norm(ped.position - ped.target)
                dist_to_recommended = np.linalg.norm(ped.position - target_exit.position)

                # 如果推荐出口更近或差距不大，更有可能响应
                if dist_to_recommended < dist_to_current * 1.3:
                    prob = 0.2  # 20%概率响应
                else:
                    prob = 0.08  # 8%概率响应

                # 根据出口拥堵情况调整概率
                _, congestion = self._compute_exit_metrics(target_exit)
                if congestion < 0.3:  # 如果推荐出口不拥堵
                    prob *= 1.5

                if np.random.random() < prob:
                    ped.target = target_exit.position.copy()

    def _check_evacuated(self):
        """检查并移除已疏散的行人

        增强版本: 同时统计按类型的疏散数量
        """
        evacuated = []
        for ped in self.sfm.pedestrians:
            for exit_obj in self.exits:
                dist = np.linalg.norm(ped.position - exit_obj.position)
                if dist < exit_obj.width:
                    evacuated.append((ped, exit_obj.name))
                    self.evacuated_count += 1
                    self.evacuated_by_exit[exit_obj.name] += 1
                    # 按类型统计
                    self.evacuated_by_type[ped.ped_type] += 1
                    break

        for ped, _ in evacuated:
            self.sfm.pedestrians.remove(ped)

        self.history['evacuated'].append(self.evacuated_count)

    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0

        # 1. 疏散奖励：每疏散一人给正奖励
        new_evacuated = self.evacuated_count - (
            self.history['evacuated'][-2] if len(self.history['evacuated']) > 1 else 0
        )
        reward += new_evacuated * 10.0

        # 2. 拥堵惩罚
        total_congestion = 0
        for exit_obj in self.exits:
            _, congestion = self._compute_exit_metrics(exit_obj)
            total_congestion += congestion
        reward -= total_congestion * 2.0
        self.history['congestion'].append(total_congestion)

        # 3. 时间惩罚（鼓励快速疏散）
        reward -= 0.1

        # 4. 完成奖励
        if len(self.sfm.pedestrians) == 0:
            reward += 100.0

        # 5. 均衡奖励：鼓励各出口分流
        counts = list(self.evacuated_by_exit.values())
        if sum(counts) > 0:
            # 计算分布均匀度（方差越小越好）
            mean_count = sum(counts) / 3
            variance = sum((c - mean_count) ** 2 for c in counts) / 3
            # 归一化方差惩罚
            balance_penalty = min(variance / 100.0, 1.0)
            reward -= balance_penalty * 0.5

        return reward

    def render(self):
        """渲染（简单版本，用于调试）"""
        if self.render_mode == "human":
            # 按出口统计
            exit_stats = f"A:{self.evacuated_by_exit['A']}, B:{self.evacuated_by_exit['B']}, C:{self.evacuated_by_exit['C']}"

            # 按类型统计
            type_stats = ", ".join([
                f"{t.value}:{c}" for t, c in self.evacuated_by_type.items() if c > 0
            ])
            if not type_stats:
                type_stats = "无"

            print(f"Step {self.current_step}: "
                  f"疏散 {self.evacuated_count}/{self.n_pedestrians} "
                  f"(出口: {exit_stats}) "
                  f"(类型: {type_stats}), "
                  f"剩余 {len(self.sfm.pedestrians)}")

    def close(self):
        """关闭环境"""
        pass
