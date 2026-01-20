"""
人群疏散强化学习环境
基于 Gymnasium 接口，用于训练 PPO/Rainbow 等算法
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

# 导入社会力模型
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sfm.social_force import SocialForceModel, Pedestrian


@dataclass
class Exit:
    """出口定义"""
    id: int
    position: np.ndarray
    width: float


class EvacuationEnv(gym.Env):
    """人群疏散环境

    智能体控制疏散引导策略，目标是最小化疏散时间和拥堵

    观测空间:
        - 各出口附近的人群密度
        - 各出口的拥堵程度
        - 剩余未疏散人数
        - 当前时间步

    动作空间:
        - 离散: 为每个区域指定推荐出口
        - 或连续: 调整各出口的吸引力权重
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        n_pedestrians: int = 50,
        scene_size: Tuple[float, float] = (30.0, 20.0),
        n_exits: int = 2,
        max_steps: int = 1000,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.n_pedestrians = n_pedestrians
        self.scene_width, self.scene_height = scene_size
        self.n_exits = n_exits
        self.max_steps = max_steps
        self.dt = dt
        self.render_mode = render_mode

        # 定义出口
        self.exits = self._create_exits()

        # 观测空间: [各出口密度(n_exits), 各出口拥堵度(n_exits), 剩余人数比例, 时间比例]
        obs_dim = self.n_exits * 2 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间: 为整体人群选择推荐出口（或调整出口吸引力）
        # 这里用离散动作：选择哪个出口作为主要引导目标
        self.action_space = spaces.Discrete(n_exits)

        # 初始化
        self.sfm = None
        self.current_step = 0
        self.evacuated_count = 0
        self.total_evacuation_time = 0

        # 记录
        self.history = {
            'evacuated': [],
            'congestion': [],
            'rewards': []
        }

    def _create_exits(self) -> List[Exit]:
        """创建出口"""
        exits = []
        # 出口均匀分布在右侧墙壁
        for i in range(self.n_exits):
            y_pos = (i + 1) * self.scene_height / (self.n_exits + 1)
            exits.append(Exit(
                id=i,
                position=np.array([self.scene_width, y_pos]),
                width=2.0
            ))
        return exits

    def _create_sfm(self) -> SocialForceModel:
        """创建社会力模型实例"""
        sfm = SocialForceModel(tau=0.5, A=2000.0, B=0.08)

        # 添加墙壁
        # 上墙
        sfm.add_obstacle(
            np.array([0, self.scene_height]),
            np.array([self.scene_width, self.scene_height])
        )
        # 下墙
        sfm.add_obstacle(np.array([0, 0]), np.array([self.scene_width, 0]))
        # 左墙
        sfm.add_obstacle(np.array([0, 0]), np.array([0, self.scene_height]))

        # 右墙（有出口的间隔）
        exit_positions = sorted([e.position[1] for e in self.exits])
        wall_segments = []

        # 从底部开始
        prev_y = 0
        for exit_obj in sorted(self.exits, key=lambda e: e.position[1]):
            exit_y = exit_obj.position[1]
            half_width = exit_obj.width / 2

            if exit_y - half_width > prev_y:
                wall_segments.append((prev_y, exit_y - half_width))
            prev_y = exit_y + half_width

        if prev_y < self.scene_height:
            wall_segments.append((prev_y, self.scene_height))

        for y_start, y_end in wall_segments:
            sfm.add_obstacle(
                np.array([self.scene_width, y_start]),
                np.array([self.scene_width, y_end])
            )

        return sfm

    def _spawn_pedestrians(self, target_exit_id: int = 0):
        """生成行人"""
        target = self.exits[target_exit_id].position

        for i in range(self.n_pedestrians):
            position = np.array([
                np.random.uniform(2, self.scene_width * 0.5),
                np.random.uniform(2, self.scene_height - 2)
            ])
            velocity = np.zeros(2)

            ped = Pedestrian(
                id=i,
                position=position,
                velocity=velocity,
                target=target.copy(),
                desired_speed=np.random.uniform(1.0, 1.5)
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

        # 生成行人（初始目标为出口0）
        self._spawn_pedestrians(target_exit_id=0)

        self.current_step = 0
        self.evacuated_count = 0
        self.total_evacuation_time = 0

        self.history = {
            'evacuated': [],
            'congestion': [],
            'rewards': []
        }

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        # 计算各出口的密度和拥堵度
        exit_densities = []
        exit_congestions = []

        for exit_obj in self.exits:
            density, congestion = self._compute_exit_metrics(exit_obj)
            exit_densities.append(density)
            exit_congestions.append(congestion)

        # 剩余人数比例
        remaining_ratio = len(self.sfm.pedestrians) / self.n_pedestrians

        # 时间比例
        time_ratio = self.current_step / self.max_steps

        obs = np.array(
            exit_densities + exit_congestions + [remaining_ratio, time_ratio],
            dtype=np.float32
        )

        return np.clip(obs, 0.0, 1.0)

    def _compute_exit_metrics(self, exit_obj: Exit) -> Tuple[float, float]:
        """计算出口附近的密度和拥堵度"""
        radius = 5.0  # 检测半径
        exit_pos = exit_obj.position

        nearby_peds = []
        for ped in self.sfm.pedestrians:
            dist = np.linalg.norm(ped.position - exit_pos)
            if dist < radius:
                nearby_peds.append(ped)

        # 密度 (归一化到 0-1)
        density = min(len(nearby_peds) / 20.0, 1.0)

        # 拥堵度 (基于平均速度下降)
        if len(nearby_peds) > 0:
            avg_speed = np.mean([ped.speed for ped in nearby_peds])
            expected_speed = 1.2  # 期望速度
            congestion = max(0, 1 - avg_speed / expected_speed)
        else:
            congestion = 0.0

        return density, congestion

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步

        Args:
            action: 选择的主要引导出口 ID

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 根据动作调整部分行人的目标出口
        self._apply_action(action)

        # 运行社会力模型多步
        for _ in range(5):  # 每个RL步骤运行5个物理步骤
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
            'step': self.current_step
        }

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int):
        """应用动作：调整行人目标"""
        target_exit = self.exits[action]

        # 只调整部分行人的目标（模拟引导信号的效果）
        for ped in self.sfm.pedestrians:
            # 根据当前位置和出口位置，有一定概率听从引导
            dist_to_target = np.linalg.norm(ped.position - target_exit.position)

            # 越近的行人越可能听从引导
            prob = np.exp(-dist_to_target / 20.0) * 0.3

            if np.random.random() < prob:
                ped.target = target_exit.position.copy()

    def _check_evacuated(self):
        """检查并移除已疏散的行人"""
        evacuated = []
        for ped in self.sfm.pedestrians:
            for exit_obj in self.exits:
                dist = np.linalg.norm(ped.position - exit_obj.position)
                if dist < exit_obj.width:
                    evacuated.append(ped)
                    self.evacuated_count += 1
                    self.total_evacuation_time += self.current_step
                    break

        for ped in evacuated:
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

        # 3. 时间惩罚（鼓励快速疏散）
        reward -= 0.1

        # 4. 完成奖励
        if len(self.sfm.pedestrians) == 0:
            reward += 100.0

        return reward

    def render(self):
        """渲染（简单版本，用于调试）"""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: "
                  f"Evacuated {self.evacuated_count}/{self.n_pedestrians}, "
                  f"Remaining {len(self.sfm.pedestrians)}")

    def close(self):
        """关闭环境"""
        pass
