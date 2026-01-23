"""
成都东站大型地铁站疏散环境
T形布局: 150m × 80m
支持紧急模式: 扶梯持续涌入 + 多出口疏散

核心特性:
- T形结构 (左侧走廊 + 上下走廊 + 中间连通区)
- 双层人流模型 (上层初始 + 下层扶梯涌入)
- 8个疏散出口 (闸机) + 3个扶梯涌入点
- 紧急模式: 电梯禁用, 扶梯只上, 闸机只出
- 带行李行人类型
- 归一化观测空间 (支持从小规模训练到大规模应用)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import yaml
from pathlib import Path

# 导入社会力模型
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sfm.social_force import (
    SocialForceModel,
    Pedestrian,
    PedestrianType,
    PEDESTRIAN_TYPE_PARAMS
)

# GPU版社会力模型
try:
    from sfm.social_force_gpu import (
        GPUSocialForceModel,
        Pedestrian as GPUPedestrian,
        PedestrianType as GPUPedestrianType,
    )
    GPU_SFM_AVAILABLE = True
except ImportError:
    GPU_SFM_AVAILABLE = False


# ========== 默认配置 ==========
DEFAULT_REWARD_WEIGHTS = {
    "evac_per_person": 15.0,
    "congestion_penalty": 4.0,
    "time_penalty": 0.3,
    "completion_bonus": 300.0,
    "balance_penalty": 1.0,
    "crush_penalty": 50.0,
    "panic_penalty": 10.0,
}

LARGE_SCALE_SAFETY = {
    "critical_density": 4.0,
    "warning_density": 2.5,
    "min_safe_distance": 0.5,
    "panic_spread_radius": 5.0,
    "panic_spread_rate": 0.1,
}

# 人流量等级
FLOW_LEVELS = {
    "small": {"upper_layer": 500, "lower_layer": 500, "spawn_rate": 3.0, "total": 1000},
    "medium": {"upper_layer": 1000, "lower_layer": 1000, "spawn_rate": 5.0, "total": 2000},
    "large": {"upper_layer": 1500, "lower_layer": 1500, "spawn_rate": 8.0, "total": 3000},
}

# 行人类型分布 (带行李)
DEFAULT_TYPE_DISTRIBUTION = {
    PedestrianType.NORMAL: 0.40,
    PedestrianType.ELDERLY: 0.10,
    PedestrianType.CHILD: 0.05,
    PedestrianType.IMPATIENT: 0.05,
    PedestrianType.WITH_SMALL_BAG: 0.25,
    PedestrianType.WITH_LUGGAGE: 0.12,
    PedestrianType.WITH_LARGE_LUGGAGE: 0.03,
}


@dataclass
class Exit:
    """出口定义 (闘機)"""
    id: str
    name: str
    position: np.ndarray
    width: float
    direction: str  # "up", "down", "left", "right"
    capacity: int = 100


@dataclass
class Escalator:
    """扶梯定义 (涌入点)"""
    id: str
    position: np.ndarray
    size: Tuple[float, float]
    direction: str = "up"
    capacity: int = 60
    spawn_point: bool = True


class LargeStationEnv(gym.Env):
    """成都东站大型地铁站疏散环境 (T形布局)

    观测空间 (41维 - 归一化):
        出口相关 (24维):
        - 8个出口密度 (8)
        - 8个出口拥堵度 (8)
        - 8个出口流量比例 (8)

        全局状态 (6维):
        - 剩余人数比例 (1)
        - 时间比例 (1)
        - 平均速度比例 (1)
        - 恐慌水平 (1)
        - 踩踏风险 (1)
        - 涌入压力 (1)

        涌入点密度 (7维):
        - 7个涌入点密度 (5扶梯 + 2步梯)

        瓶颈区域 (4维):
        - 4个关键瓶颈区域密度

    动作空间: Discrete(8) - 选择推荐的疏散出口

    T形结构:
        - 左侧走廊: X=0-20, Y=0-80
        - 上部走廊: X=20-150, Y=55-70
        - 下部走廊: X=20-150, Y=10-25
        - 中间连通区: X=20-150, Y=25-55
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        flow_level: str = "medium",
        scene_size: Tuple[float, float] = (150.0, 80.0),
        max_steps: int = 6000,
        dt: float = 0.1,
        render_mode: Optional[str] = None,
        type_distribution: Optional[Dict[PedestrianType, float]] = None,
        enable_enhanced_behaviors: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
        use_gpu_sfm: bool = False,
        sfm_device: str = "auto",
        emergency_mode: bool = True,
        config_path: Optional[str] = None,
    ):
        super().__init__()

        # 加载配置
        self.config = self._load_config(config_path)

        # 基本参数
        self.scene_width, self.scene_height = scene_size
        self.max_steps = max_steps
        self.dt = dt
        self.render_mode = render_mode
        self.emergency_mode = emergency_mode

        # 人流量配置
        self.flow_level = flow_level
        flow_config = FLOW_LEVELS.get(flow_level, FLOW_LEVELS["medium"])
        self.upper_layer_count = flow_config["upper_layer"]
        self.lower_layer_count = flow_config["lower_layer"]
        self.spawn_rate = flow_config["spawn_rate"]
        self.n_pedestrians = flow_config["total"]

        # 行人类型分布
        self.type_distribution = type_distribution or DEFAULT_TYPE_DISTRIBUTION
        self.enable_enhanced_behaviors = enable_enhanced_behaviors

        # 奖励权重
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}

        # GPU配置
        self.use_gpu_sfm = use_gpu_sfm and GPU_SFM_AVAILABLE
        self.sfm_device = sfm_device

        # 初始化场景设施
        self._init_exits()
        self._init_escalators()

        # 观测空间 (41维) - 按文档5.3节设计
        # 8出口×3=24 + 全局6 + 涌入点7 + 瓶颈4 = 41
        obs_dim = 41
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间
        self.action_space = spaces.Discrete(len(self.exits))

        # 状态变量
        self.sfm = None
        self.current_step = 0
        self.evacuated_count = 0
        self.evacuated_by_exit = {exit.id: 0 for exit in self.exits}
        self.spawned_from_lower = 0
        self.evacuation_rate_buffer = [0.0] * 5
        self.last_evacuated_count = 0

        # 历史记录
        self.history = {
            'evacuated': [],
            'congestion': [],
            'rewards': [],
            'inflow': [],
        }

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path is None:
            default_path = Path(__file__).parent.parent.parent / "configs" / "large_station_config.yaml"
            if default_path.exists():
                config_path = str(default_path)
            else:
                return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return {}

    def _init_exits(self):
        """初始化出口 (8个闘機)"""
        self.exits = [
            # 左侧闘機a (Y=40中心)
            Exit("gate_a", "闸机a", np.array([0.0, 40.0]), 20.0, "left", 100),
            # 右侧闘機子 (Y=40中心)
            Exit("gate_zi", "闸机子", np.array([150.0, 40.0]), 20.0, "right", 100),
            # 上排闘機 (Y=70)
            Exit("gate_b", "闸机b", np.array([52.5, 70.0]), 15.0, "up", 150),
            Exit("gate_c", "闸机c", np.array([82.5, 70.0]), 15.0, "up", 150),
            Exit("gate_d", "闸机d", np.array([112.5, 70.0]), 15.0, "up", 150),
            # 下排闘機 (Y=10)
            Exit("gate_e", "闸机e", np.array([52.5, 10.0]), 15.0, "down", 150),
            Exit("gate_f", "闸机f", np.array([82.5, 10.0]), 15.0, "down", 150),
            Exit("gate_g", "闸机g", np.array([112.5, 10.0]), 15.0, "down", 150),
        ]
        self.n_exits = len(self.exits)
        self.exit_names = [e.name for e in self.exits]

    def _init_escalators(self):
        """初始化涌入点 (7个: 5扶梯 + 2步梯)"""
        self.escalators = [
            # 中间3个扶梯
            Escalator("escalator_1", np.array([52.5, 40.0]), (25.0, 16.0), "up", 60, True),
            Escalator("escalator_2", np.array([82.5, 40.0]), (25.0, 16.0), "up", 60, True),
            Escalator("escalator_3", np.array([112.5, 40.0]), (25.0, 16.0), "up", 60, True),
            # 左上扶梯和步梯
            Escalator("escalator_left_upper", np.array([10.0, 72.0]), (6.0, 12.0), "up", 60, True),
            Escalator("stairs_left_upper", np.array([2.0, 72.0]), (6.0, 12.0), "up", 40, True),
            # 左下扶梯和步梯
            Escalator("escalator_left_lower", np.array([10.0, 8.0]), (6.0, 12.0), "up", 60, True),
            Escalator("stairs_left_lower", np.array([2.0, 8.0]), (6.0, 12.0), "up", 40, True),
        ]
        self.n_escalators = len(self.escalators)

    def _create_sfm(self):
        """创建社会力模型实例"""
        sfm_params = dict(
            tau=0.5,
            A=2000.0,
            B=0.08,
            wall_A=5000.0,
            wall_B=0.1,
            enable_waiting=self.enable_enhanced_behaviors,
            enable_perturbation=self.enable_enhanced_behaviors,
            enable_panic=self.enable_enhanced_behaviors,
        )

        if self.use_gpu_sfm:
            sfm = GPUSocialForceModel(device=self.sfm_device, **sfm_params)
        else:
            sfm = SocialForceModel(**sfm_params)

        # 添加T形边界墙
        self._add_t_shape_walls(sfm)
        # 添加扶梯护栏
        self._add_escalator_barriers(sfm)

        return sfm

    def _add_t_shape_walls(self, sfm):
        """添加T形边界墙 (有出口间隔)"""
        # T形结构:
        # - 左侧走廊: X=0-20, Y=0-80
        # - 上部走廊: X=20-150, Y=55-70
        # - 下部走廊: X=20-150, Y=10-25
        # - 中间连通区: X=20-150, Y=25-55

        # ===== 左侧走廊 =====
        # 左墙 (有闸机a间隔: Y=30-50)
        sfm.add_obstacle(np.array([0, 0]), np.array([0, 30]))
        sfm.add_obstacle(np.array([0, 50]), np.array([0, 80]))
        # 顶边
        sfm.add_obstacle(np.array([0, 80]), np.array([20, 80]))
        # 底边
        sfm.add_obstacle(np.array([0, 0]), np.array([20, 0]))

        # ===== 上部走廊 =====
        # 上边 (有闸机b/c/d间隔)
        # 闸机b: X=45-60, 闸机c: X=75-90, 闸机d: X=105-120
        sfm.add_obstacle(np.array([20, 70]), np.array([45, 70]))
        sfm.add_obstacle(np.array([60, 70]), np.array([75, 70]))
        sfm.add_obstacle(np.array([90, 70]), np.array([105, 70]))
        sfm.add_obstacle(np.array([120, 70]), np.array([150, 70]))
        # 右上角垂直墙
        sfm.add_obstacle(np.array([150, 70]), np.array([150, 55]))
        # 上走廊下边 (与中间区连通, 无墙)

        # ===== 下部走廊 =====
        # 下边 (有闸机e/f/g间隔)
        sfm.add_obstacle(np.array([20, 10]), np.array([45, 10]))
        sfm.add_obstacle(np.array([60, 10]), np.array([75, 10]))
        sfm.add_obstacle(np.array([90, 10]), np.array([105, 10]))
        sfm.add_obstacle(np.array([120, 10]), np.array([150, 10]))
        # 右下角垂直墙
        sfm.add_obstacle(np.array([150, 10]), np.array([150, 25]))

        # ===== 中间区域右墙 =====
        # 右墙 (有闸机子间隔: Y=30-50)
        sfm.add_obstacle(np.array([150, 25]), np.array([150, 30]))
        sfm.add_obstacle(np.array([150, 50]), np.array([150, 55]))

        # ===== 左侧走廊与中间区域连接处 =====
        # 上部连接 (Y=70到Y=80之间)
        sfm.add_obstacle(np.array([20, 70]), np.array([20, 80]))
        # 下部连接 (Y=0到Y=10之间)
        sfm.add_obstacle(np.array([20, 0]), np.array([20, 10]))

    def _add_escalator_barriers(self, sfm):
        """添加扶梯护栏"""
        for esc in self.escalators:
            x, y = esc.position
            w, h = esc.size

            # 扶梯两侧护栏
            sfm.add_obstacle(np.array([x - w/2, y - h/2]), np.array([x - w/2, y + h/2]))
            sfm.add_obstacle(np.array([x + w/2, y - h/2]), np.array([x + w/2, y + h/2]))

    def _is_in_t_shape(self, x: float, y: float) -> bool:
        """检查位置是否在T形区域内"""
        # 左侧走廊
        if 0 <= x <= 20 and 0 <= y <= 80:
            return True
        # 上部走廊
        if 20 <= x <= 150 and 55 <= y <= 70:
            return True
        # 下部走廊
        if 20 <= x <= 150 and 10 <= y <= 25:
            return True
        # 中间连通区
        if 20 <= x <= 150 and 25 <= y <= 55:
            return True
        return False

    def _spawn_upper_layer(self):
        """生成上层初始行人"""
        types = list(self.type_distribution.keys())
        probs = list(self.type_distribution.values())
        total = sum(probs)
        probs = [p / total for p in probs]

        # T形区域内的生成区
        spawn_areas = [
            {"bounds": (25, 55, 145, 68), "weight": 0.3},   # 上部走廊
            {"bounds": (25, 12, 145, 23), "weight": 0.3},   # 下部走廊
            {"bounds": (25, 28, 145, 52), "weight": 0.3},   # 中间区域
            {"bounds": (2, 20, 18, 60), "weight": 0.1},     # 左侧走廊
        ]

        for i in range(self.upper_layer_count):
            # 选择生成区域
            area_weights = [a["weight"] for a in spawn_areas]
            area_idx = np.random.choice(len(spawn_areas), p=area_weights)
            x_min, y_min, x_max, y_max = spawn_areas[area_idx]["bounds"]

            position = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])
            velocity = np.zeros(2)

            initial_target = self._select_initial_target(position)
            ped_type = np.random.choice(types, p=probs)

            ped = Pedestrian.create_with_type(
                id=i,
                position=position,
                velocity=velocity,
                target=initial_target,
                ped_type=ped_type,
                speed_variation=True
            )
            self.sfm.add_pedestrian(ped)

    def _select_initial_target(self, position: np.ndarray) -> np.ndarray:
        """根据位置智能选择初始目标出口"""
        x, y = position

        # 根据位置选择最近的出口群
        if y > 55:
            # 上部走廊 - 偏向上排出口
            candidates = [2, 3, 4]  # gate_b/c/d
        elif y < 25:
            # 下部走廊 - 偏向下排出口
            candidates = [5, 6, 7]  # gate_e/f/g
        elif x < 20:
            # 左侧走廊 - 偏向左侧出口
            candidates = [0]  # gate_a
        else:
            # 中间区域 - 根据位置选择
            if y > 40:
                candidates = [2, 3, 4]  # 上排
            else:
                candidates = [5, 6, 7]  # 下排

        # 根据距离加权选择
        weights = []
        for idx in candidates:
            dist = np.linalg.norm(position - self.exits[idx].position)
            weights.append(1.0 / (dist + 1.0))

        total = sum(weights)
        probs = [w / total for w in weights]
        choice = np.random.choice(candidates, p=probs)

        return self.exits[choice].position.copy()

    def _spawn_from_escalator(self):
        """从扶梯生成涌入行人"""
        if self.spawned_from_lower >= self.lower_layer_count:
            return 0

        spawn_this_step = self.spawn_rate * self.dt
        spawn_count = np.random.poisson(spawn_this_step)
        spawn_count = min(spawn_count, self.lower_layer_count - self.spawned_from_lower)

        if spawn_count <= 0:
            return 0

        types = list(self.type_distribution.keys())
        probs = list(self.type_distribution.values())
        total = sum(probs)
        probs = [p / total for p in probs]

        spawned = 0
        next_id = len(self.sfm.pedestrians)

        for _ in range(spawn_count):
            esc = np.random.choice(self.escalators)

            position = np.array([
                esc.position[0] + np.random.uniform(-2, 2),
                esc.position[1] + np.random.uniform(-2, 2)
            ])
            velocity = np.zeros(2)

            target = self._select_initial_target(position)
            ped_type = np.random.choice(types, p=probs)

            ped = Pedestrian.create_with_type(
                id=next_id,
                position=position,
                velocity=velocity,
                target=target,
                ped_type=ped_type,
                speed_variation=True
            )
            self.sfm.add_pedestrian(ped)

            next_id += 1
            spawned += 1
            self.spawned_from_lower += 1

        return spawned

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        self.sfm = self._create_sfm()
        self._spawn_upper_layer()

        self.current_step = 0
        self.evacuated_count = 0
        self.evacuated_by_exit = {exit.id: 0 for exit in self.exits}
        self.spawned_from_lower = 0
        self.evacuation_rate_buffer = [0.0] * 5
        self.last_evacuated_count = 0

        self.history = {
            'evacuated': [],
            'congestion': [],
            'rewards': [],
            'inflow': [],
        }

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """获取观测 (41维) - 按文档5.3节设计

        [0-7]   8个出口密度 (归一化0-1)
        [8-15]  8个出口拥堵度 (归一化0-1)
        [16-23] 8个出口流量比例 (归一化0-1)
        [24]    剩余人数比例
        [25]    时间比例
        [26]    平均速度比例
        [27]    恐慌水平
        [28]    踩踏风险
        [29]    涌入压力
        [30-36] 7个涌入点密度 (5扶梯+2步梯)
        [37-40] 4个关键瓶颈区域密度
        """
        total_peds = len(self.sfm.pedestrians)

        # === 出口相关 (24维) ===
        exit_densities = []
        exit_congestions = []
        exit_flow_ratios = []

        for exit_obj in self.exits:
            density, congestion = self._compute_exit_metrics(exit_obj)
            exit_densities.append(density)
            exit_congestions.append(congestion)

            flow_count = sum(
                1 for ped in self.sfm.pedestrians
                if np.linalg.norm(ped.target - exit_obj.position) < 5.0
            )
            flow_ratio = flow_count / max(total_peds, 1)
            exit_flow_ratios.append(flow_ratio)

        # === 全局状态 (6维) ===
        remaining_ratio = total_peds / max(self.n_pedestrians, 1)
        time_ratio = self.current_step / self.max_steps

        # 平均速度比例
        if total_peds > 0:
            avg_speed = np.mean([ped.speed for ped in self.sfm.pedestrians])
            avg_speed_ratio = min(avg_speed / 1.5, 1.0)  # 1.5 m/s为参考速度
        else:
            avg_speed_ratio = 1.0

        # 恐慌水平
        if total_peds > 0:
            panic_level = np.mean([ped.panic_factor for ped in self.sfm.pedestrians])
        else:
            panic_level = 0.0

        # 踩踏风险
        max_density, _ = self._detect_crush_risk()
        crush_risk = min(max_density / LARGE_SCALE_SAFETY["critical_density"], 1.0)

        # 涌入压力 (剩余待涌入人数比例)
        remaining_spawn = self.lower_layer_count - self.spawned_from_lower
        spawn_pressure = remaining_spawn / max(self.lower_layer_count, 1)

        # === 涌入点密度 (7维) ===
        escalator_densities = self._compute_escalator_densities()

        # === 瓶颈区域密度 (4维) ===
        bottleneck_densities = self._compute_bottleneck_densities()

        obs = np.array(
            exit_densities +           # 8维
            exit_congestions +         # 8维
            exit_flow_ratios +         # 8维
            [remaining_ratio,          # 1维
             time_ratio,               # 1维
             avg_speed_ratio,          # 1维
             panic_level,              # 1维
             crush_risk,               # 1维
             spawn_pressure] +         # 1维
            escalator_densities +      # 7维
            bottleneck_densities,      # 4维
            dtype=np.float32
        )

        return np.clip(obs, 0.0, 1.0)

    def _compute_exit_metrics(self, exit_obj: Exit) -> Tuple[float, float]:
        """计算出口密度和拥堵度"""
        radius = 10.0
        exit_pos = exit_obj.position

        nearby_peds = [
            ped for ped in self.sfm.pedestrians
            if np.linalg.norm(ped.position - exit_pos) < radius
        ]

        max_density_people = max(self.n_pedestrians / self.n_exits, 20.0)
        density = min(len(nearby_peds) / max_density_people, 1.0)

        if len(nearby_peds) > 0:
            avg_speed = np.mean([ped.speed for ped in nearby_peds])
            expected_speed = 1.2
            congestion = max(0, 1 - avg_speed / expected_speed)
        else:
            congestion = 0.0

        return density, congestion

    def _compute_escalator_densities(self) -> List[float]:
        """计算7个涌入点密度 (5扶梯 + 2步梯)"""
        densities = []
        detection_radius = 8.0
        max_density_people = max(self.n_pedestrians * 0.10, 10.0)

        for esc in self.escalators:
            count = sum(
                1 for ped in self.sfm.pedestrians
                if np.linalg.norm(ped.position - esc.position) < detection_radius
            )
            density = min(count / max_density_people, 1.0)
            densities.append(density)

        return densities

    def _compute_bottleneck_densities(self) -> List[float]:
        """计算4个关键瓶颈区域密度"""
        # 4个关键瓶颈区域：
        # 1. 左侧走廊中部 (闸机a前)
        # 2. 上部走廊中部
        # 3. 下部走廊中部
        # 4. 中间区域核心
        bottleneck_areas = [
            np.array([10.0, 40.0]),    # 左侧走廊中部
            np.array([85.0, 62.0]),    # 上部走廊中部
            np.array([85.0, 18.0]),    # 下部走廊中部
            np.array([85.0, 40.0]),    # 中间区域核心
        ]

        densities = []
        detection_radius = 10.0
        max_density_people = max(self.n_pedestrians * 0.15, 15.0)

        for area_center in bottleneck_areas:
            count = sum(
                1 for ped in self.sfm.pedestrians
                if np.linalg.norm(ped.position - area_center) < detection_radius
            )
            density = min(count / max_density_people, 1.0)
            densities.append(density)

        return densities

    def _detect_crush_risk(self) -> Tuple[float, int]:
        """检测踩踏风险"""
        detection_radius = 2.0
        critical_density = LARGE_SCALE_SAFETY["critical_density"]
        max_density = 0.0
        danger_count = 0

        for ped in self.sfm.pedestrians:
            nearby_count = sum(
                1 for other in self.sfm.pedestrians
                if np.linalg.norm(ped.position - other.position) < detection_radius
                and other.id != ped.id
            )
            local_density = nearby_count / (np.pi * detection_radius ** 2)

            if local_density > max_density:
                max_density = local_density

            if local_density > critical_density:
                danger_count += 1

        return max_density, danger_count

    def _spread_panic(self) -> int:
        """恐慌传播"""
        spread_radius = LARGE_SCALE_SAFETY["panic_spread_radius"]
        spread_rate = LARGE_SCALE_SAFETY["panic_spread_rate"]
        affected_count = 0

        panicked_peds = [
            ped for ped in self.sfm.pedestrians
            if ped.panic_factor > 0.3
        ]

        for source in panicked_peds:
            for target in self.sfm.pedestrians:
                if target.id == source.id:
                    continue

                dist = np.linalg.norm(target.position - source.position)
                if dist < spread_radius:
                    spread_strength = spread_rate * (1 - dist / spread_radius)
                    old_panic = target.panic_factor
                    target.panic_factor = min(1.0, target.panic_factor + spread_strength * source.panic_factor)
                    if target.panic_factor > old_panic:
                        affected_count += 1

        return affected_count

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步仿真"""
        self.current_step += 1

        if self.emergency_mode:
            inflow_count = self._spawn_from_escalator()
            self.history['inflow'].append(inflow_count)

        self._apply_guidance(action)
        self.sfm.step(self.dt)
        new_evacuated = self._process_evacuation()

        max_density, danger_count = self._detect_crush_risk()
        panic_affected = self._spread_panic()

        reward = self._compute_reward(new_evacuated, danger_count, panic_affected)

        self.history['evacuated'].append(new_evacuated)
        self.history['rewards'].append(reward)

        evacuation_ratio = new_evacuated / max(self.n_pedestrians, 1)
        self.evacuation_rate_buffer.pop(0)
        self.evacuation_rate_buffer.append(evacuation_ratio)

        all_evacuated = (
            len(self.sfm.pedestrians) == 0 and
            self.spawned_from_lower >= self.lower_layer_count
        )
        terminated = all_evacuated
        truncated = self.current_step >= self.max_steps

        info = {
            'evacuated': self.evacuated_count,
            'evacuated_by_exit': self.evacuated_by_exit.copy(),
            'remaining': len(self.sfm.pedestrians),
            'spawned_from_lower': self.spawned_from_lower,
            'max_density': max_density,
            'danger_count': danger_count,
            'panic_affected': panic_affected,
            'step': self.current_step,
            'time': self.current_step * self.dt,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _apply_guidance(self, action: int):
        """应用PPO引导策略"""
        if action < 0 or action >= len(self.exits):
            return

        recommended_exit = self.exits[action]

        for ped in self.sfm.pedestrians:
            if ped.guidance_count >= 2:
                continue

            dist_to_target = np.linalg.norm(ped.position - ped.target)
            if dist_to_target < 10.0:
                continue

            current_exit_idx = self._find_nearest_exit(ped.target)
            if current_exit_idx < 0:
                continue

            current_density, current_congestion = self._compute_exit_metrics(self.exits[current_exit_idx])
            rec_density, rec_congestion = self._compute_exit_metrics(recommended_exit)

            if rec_congestion < current_congestion - 0.1:
                ped.target = recommended_exit.position.copy()
                ped.guidance_count += 1

    def _find_nearest_exit(self, position: np.ndarray) -> int:
        """找到最近的出口索引"""
        min_dist = float('inf')
        nearest_idx = -1

        for idx, exit_obj in enumerate(self.exits):
            dist = np.linalg.norm(position - exit_obj.position)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

        return nearest_idx

    def _process_evacuation(self) -> int:
        """处理疏散"""
        evacuated_this_step = 0
        evacuation_radius = 3.0

        peds_to_remove = []

        for ped in self.sfm.pedestrians:
            for exit_obj in self.exits:
                dist = np.linalg.norm(ped.position - exit_obj.position)
                if dist < evacuation_radius:
                    peds_to_remove.append(ped)
                    self.evacuated_by_exit[exit_obj.id] += 1
                    evacuated_this_step += 1
                    break

        for ped in peds_to_remove:
            self.sfm.pedestrians.remove(ped)

        self.evacuated_count += evacuated_this_step
        return evacuated_this_step

    def _compute_reward(
        self,
        new_evacuated: int,
        danger_count: int,
        panic_affected: int
    ) -> float:
        """计算奖励"""
        reward = 0.0

        evacuation_ratio = new_evacuated / max(self.n_pedestrians, 1)
        reward += evacuation_ratio * self.reward_weights["evac_per_person"] * 100

        total_congestion = sum(
            self._compute_exit_metrics(exit_obj)[1]
            for exit_obj in self.exits
        )
        avg_congestion = total_congestion / self.n_exits
        reward -= avg_congestion * self.reward_weights["congestion_penalty"]

        reward -= self.reward_weights["time_penalty"]

        if danger_count > 0:
            crush_penalty = (danger_count / max(len(self.sfm.pedestrians), 1)) * self.reward_weights["crush_penalty"]
            reward -= crush_penalty

        if panic_affected > 0:
            panic_penalty = (panic_affected / max(len(self.sfm.pedestrians), 1)) * self.reward_weights["panic_penalty"]
            reward -= panic_penalty

        all_evacuated = (
            len(self.sfm.pedestrians) == 0 and
            self.spawned_from_lower >= self.lower_layer_count
        )
        if all_evacuated:
            time_bonus = max(0, (self.max_steps - self.current_step) / self.max_steps)
            reward += self.reward_weights["completion_bonus"] * (1 + time_bonus)

        return reward

    def render(self):
        """渲染"""
        pass

    def close(self):
        """关闭环境"""
        pass
