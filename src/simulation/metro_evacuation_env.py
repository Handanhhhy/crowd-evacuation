"""
成都东客站地铁出站口强化学习环境
适配3出口、闸机、柱子的复杂场景

增强版本:
- 支持多种行人类型 (老人、儿童、急躁型等)
- 可选GBM行为预测修正
- 更真实的行人行为 (等待、犹豫、恐慌)
- Social-LSTM神经网络轨迹预测 (预测性疏通系统)
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

# GPU版社会力模型
try:
    from sfm.social_force_gpu import (
        GPUSocialForceModel,
        Pedestrian as GPUPedestrian,
        PedestrianType as GPUPedestrianType,
        PEDESTRIAN_TYPE_PARAMS as GPU_PEDESTRIAN_TYPE_PARAMS
    )
    GPU_SFM_AVAILABLE = True
except ImportError:
    GPU_SFM_AVAILABLE = False
    print("警告: GPU版SFM不可用，将使用CPU版本")

# 尝试导入轨迹预测器
try:
    from ml.trajectory_predictor import TrajectoryPredictor
    TRAJECTORY_PREDICTOR_AVAILABLE = True
except ImportError:
    TRAJECTORY_PREDICTOR_AVAILABLE = False
    print("警告: 轨迹预测器不可用，使用线性外推")


# ========== 分层预测式引导系统配置 ==========
GUIDANCE_CONFIG = {
    'max_guidance_count': 2,           # 每人最多被引导2次
    'cooldown_time': 6.0,              # 冷却时间6秒
    'min_distance_to_target': 6.0,     # 距目标>6米才可引导
    'guidance_zone_x': 20.0,           # x>20进入引导区
    'problem_prediction_horizon': 12,  # 预测12步(4.8秒)
    'congestion_threshold': 0.35,      # 拥堵度阈值
    'corner_trap_radius': 3.5,         # 角落陷阱检测半径
    # 主动分流参数
    'exit_imbalance_threshold': 0.40,  # 出口负载不均阈值（>40%则分流）
    'min_peds_for_rebalance': 8,       # 至少8人时才考虑分流
    'rebalance_distance_threshold': 10.0,  # 距出口>10米才分流
    # 引导收益判断（重要：考虑拥堵收益）
    'min_distance_benefit': 0.0,       # 不要求距离更近（允许为了避堵走远路）
    'min_congestion_benefit': 0.1,     # 拥堵度差异>0.1就值得改道
}


REWARD_DEFAULTS = {
    "evac_per_person": 12.0,          # 每人疏散奖励 (增加)
    "congestion_penalty": 3.0,         # 拥堵惩罚 (增加)
    "time_penalty": 0.2,               # 时间惩罚 (增加)
    "completion_bonus": 200.0,         # 完成奖励 (增加)
    "balance_penalty": 0.8,            # 均衡惩罚 (增加)
    # 新增奖励项
    "flow_efficiency_bonus": 1.5,      # 人流效率奖励
    "safety_distance_bonus": 0.5,      # 安全间距奖励
    "guidance_penalty": 0.3,           # 频繁引导惩罚
    "evacuation_rate_bonus": 2.0,      # 疏散速率提升奖励
    "crush_penalty": 10.0,             # 踩踏风险惩罚 (大规模安全)
}

# 大规模场景安全阈值
LARGE_SCALE_SAFETY = {
    "critical_density": 4.0,           # 危险密度阈值 (人/m²) - 超过可能踩踏
    "warning_density": 2.5,            # 警告密度阈值 (人/m²)
    "min_safe_distance": 0.5,          # 最小安全距离 (米)
    "panic_spread_radius": 5.0,        # 恐慌传播半径 (米)
    "panic_spread_rate": 0.1,          # 恐慌传播速率
}

@dataclass
class Exit:
    """出口定义"""
    id: int
    name: str
    position: np.ndarray
    width: float


class MetroEvacuationEnv(gym.Env):
    """成都东客站地铁出站口疏散环境

    观测空间 (16维 - 增强版):
        - 3个出口的密度 (3)
        - 3个出口的拥堵度 (3)
        - 3个出口的人流方向占比 (3) - 新增
        - 3个历史疏散速率 (3) - 新增 (最近3步的疏散人数)
        - 瓶颈点密度 (2) - 新增 (闸机区/柱子区)
        - 剩余人数比例 (1)
        - 时间比例 (1)

    动作空间: Discrete(3) - 选择推荐出口A/B/C

    奖励函数 (增强版):
        - 疏散奖励: +12 × 新疏散人数
        - 拥堵惩罚: -3 × 总拥堵度
        - 时间惩罚: -0.2/步
        - 完成奖励: +200
        - 均衡奖励: 鼓励各出口分流
        - 人流效率奖励: 鼓励高效疏散 (新增)
        - 安全间距奖励: 避免过度拥挤 (新增)
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
        # 神经网络轨迹预测参数
        trajectory_model_path: Optional[str] = None,
        enable_neural_prediction: bool = True,
        trajectory_device: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        # GPU加速SFM参数
        use_gpu_sfm: bool = False,
        sfm_device: str = "auto",
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
            trajectory_model_path: Social-LSTM轨迹预测模型路径 (可选)
            enable_neural_prediction: 是否启用神经网络轨迹预测
            trajectory_device: 轨迹预测设备 ('auto'/'cpu'/'cuda'/'mps')
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

        # 神经网络轨迹预测器 (Social-LSTM)
        self.trajectory_predictor = None
        self.enable_neural_prediction = enable_neural_prediction
        self.trajectory_model_path = trajectory_model_path
        self.trajectory_device = trajectory_device or "auto"
        if enable_neural_prediction and TRAJECTORY_PREDICTOR_AVAILABLE:
            self._load_trajectory_predictor(trajectory_model_path)

        # GPU加速SFM配置
        self.use_gpu_sfm = use_gpu_sfm and GPU_SFM_AVAILABLE
        self.sfm_device = sfm_device
        if use_gpu_sfm and not GPU_SFM_AVAILABLE:
            print("警告: 请求使用GPU SFM但不可用，将使用CPU版本")

        self.reward_weights = {**REWARD_DEFAULTS, **(reward_weights or {})}

        # 角落陷阱位置 (容易卡住的地方)
        self.corner_traps = [
            np.array([60, 40]),   # Exit C 上方右角落
            np.array([60, 0]),    # 右下角落
            np.array([0, 40]),    # 左上角落
            np.array([0, 0]),     # 左下角落
            np.array([55, 16]),   # 楼梯旁边
            np.array([55, 24]),   # 楼梯旁边
        ]

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

        # 增强观测空间: 16维
        # [3出口密度, 3出口拥堵度, 3人流方向占比, 3历史疏散速率, 2瓶颈密度, 剩余比例, 时间比例]
        obs_dim = 16
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # 历史疏散速率缓冲区 (最近3步)
        self.evacuation_rate_buffer = [0.0, 0.0, 0.0]
        self.last_evacuated_count = 0

        # 动作空间: 选择推荐出口 (A/B/C)
        self.action_space = spaces.Discrete(self.n_exits)

        # 初始化
        self.sfm = None
        self.current_step = 0
        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.evacuated_by_type = {t: 0 for t in PedestrianType}  # 按类型统计

        # 分层预测式引导系统状态
        self.last_action = 0  # 上一次PPO动作（推荐出口）

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

    def _load_trajectory_predictor(self, model_path: Optional[str] = None) -> bool:
        """加载Social-LSTM轨迹预测模型

        Args:
            model_path: 模型文件路径 (None则自动查找默认路径)

        Returns:
            是否加载成功
        """
        if not TRAJECTORY_PREDICTOR_AVAILABLE:
            print("轨迹预测器模块不可用")
            return False

        # 如果没有指定路径，自动查找默认模型路径
        if model_path is None:
            default_paths = [
                Path(__file__).parent.parent.parent / "outputs" / "models" / "social_lstm.pt",
                Path("outputs/models/social_lstm.pt"),
            ]
            for p in default_paths:
                if p.exists():
                    model_path = str(p.absolute())
                    break

        try:
            self.trajectory_predictor = TrajectoryPredictor(
                model_path=model_path,
                obs_len=8,
                pred_len=12,
                device=self.trajectory_device
            )
            # 立即触发模型加载
            self.trajectory_predictor._ensure_model_loaded()
            mode = '神经网络' if self.trajectory_predictor.use_neural_network else '线性外推'
            print(f"轨迹预测器已加载 (模式: {mode})")
            return True
        except Exception as e:
            print(f"轨迹预测器加载失败: {e}")
            self.trajectory_predictor = None
            return False

    def _create_sfm(self):
        """创建社会力模型实例（包含地铁站场景的障碍物）

        如果启用增强行为，将开启等待、犹豫、恐慌等特性
        支持CPU和GPU(MPS/CUDA)两种模式
        """
        sfm_params = dict(
            tau=0.5,
            A=2000.0,
            B=0.08,
            wall_A=5000.0,
            wall_B=0.1,
            enable_waiting=self.enable_enhanced_behaviors,
            enable_perturbation=self.enable_enhanced_behaviors,
            enable_panic=self.enable_enhanced_behaviors,
            waiting_density_threshold=0.8,
            perturbation_sigma=0.05,
            panic_density_threshold=1.5,
            gbm_predictor=self.gbm_predictor,
            gbm_weight=0.3,
        )

        if self.use_gpu_sfm:
            sfm = GPUSocialForceModel(device=self.sfm_device, **sfm_params)
        else:
            sfm = SocialForceModel(**sfm_params)

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
        self.last_action = 0  # 重置引导系统状态

        # 重置增强观测相关状态
        self.evacuation_rate_buffer = [0.0, 0.0, 0.0]
        self.last_evacuated_count = 0
        self.guidance_count_this_episode = 0

        self.history = {
            'evacuated': [],
            'congestion': [],
            'rewards': []
        }

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """获取增强观测 (16维)

        观测向量构成:
        [0-2]  3个出口密度
        [3-5]  3个出口拥堵度
        [6-8]  3个出口人流方向占比 (走向该出口的行人比例)
        [9-11] 3个历史疏散速率 (最近3步疏散人数/总人数)
        [12-13] 2个瓶颈点密度 (闸机区、柱子区)
        [14]   剩余人数比例
        [15]   时间比例
        """
        exit_densities = []
        exit_congestions = []
        exit_flow_ratios = []

        total_peds = len(self.sfm.pedestrians)

        for exit_obj in self.exits:
            density, congestion = self._compute_exit_metrics(exit_obj)
            exit_densities.append(density)
            exit_congestions.append(congestion)

            # 计算走向该出口的行人比例
            flow_count = 0
            for ped in self.sfm.pedestrians:
                if np.linalg.norm(ped.target - exit_obj.position) < 2.0:
                    flow_count += 1
            flow_ratio = flow_count / max(total_peds, 1)
            exit_flow_ratios.append(flow_ratio)

        # 历史疏散速率 (已归一化到0-1)
        # buffer 存储的是疏散比例，最大合理值约0.1 (每步疏散10%的人)
        # 乘以10使其映射到0-1范围
        evacuation_rates = [min(r * 10.0, 1.0) for r in self.evacuation_rate_buffer]

        # 瓶颈点密度
        bottleneck_densities = self._compute_bottleneck_densities()

        # 剩余人数比例
        remaining_ratio = total_peds / max(self.n_pedestrians, 1)

        # 时间比例
        time_ratio = self.current_step / self.max_steps

        obs = np.array(
            exit_densities +           # [0-2]
            exit_congestions +         # [3-5]
            exit_flow_ratios +         # [6-8]
            evacuation_rates +         # [9-11]
            bottleneck_densities +     # [12-13]
            [remaining_ratio, time_ratio],  # [14-15]
            dtype=np.float32
        )

        return np.clip(obs, 0.0, 1.0)

    def _compute_bottleneck_densities(self) -> List[float]:
        """计算瓶颈点密度

        瓶颈区域:
        1. 闸机区 (x: 18-22, y: 7-33) - 主要拥堵点
        2. 柱子区 (围绕6个柱子的区域) - 次要拥堵点

        动态归一化: 基准随总人数调整，使模型能从小规模泛化到大规模

        Returns:
            [闸机区密度, 柱子区密度] (归一化到0-1)
        """
        gate_zone = {'x_min': 18, 'x_max': 22, 'y_min': 7, 'y_max': 33}
        gate_count = 0
        pillar_count = 0

        for ped in self.sfm.pedestrians:
            x, y = ped.position

            # 检查是否在闸机区
            if (gate_zone['x_min'] <= x <= gate_zone['x_max'] and
                gate_zone['y_min'] <= y <= gate_zone['y_max']):
                gate_count += 1

            # 检查是否靠近任何柱子 (3米内)
            for pillar in self.pillars:
                if np.linalg.norm(ped.position - pillar) < 3.0:
                    pillar_count += 1
                    break

        # 动态归一化 (基准随人数调整)
        # 闸机区最多容纳约25%的人，柱子区约25%的人
        max_gate_people = max(self.n_pedestrians * 0.25, 5.0)
        max_pillar_people = max(self.n_pedestrians * 0.25, 5.0)

        gate_density = min(gate_count / max_gate_people, 1.0)
        pillar_density = min(pillar_count / max_pillar_people, 1.0)

        return [gate_density, pillar_density]

    def _compute_exit_metrics(self, exit_obj: Exit) -> Tuple[float, float]:
        """计算出口附近的密度和拥堵度

        动态归一化: 密度基准随总人数调整，使模型能从小规模泛化到大规模
        """
        radius = 8.0  # 检测半径（地铁站场景更大）
        exit_pos = exit_obj.position

        nearby_peds = []
        for ped in self.sfm.pedestrians:
            dist = np.linalg.norm(ped.position - exit_pos)
            if dist < radius:
                nearby_peds.append(ped)

        # 密度 (动态归一化到 0-1)
        # 基准: 假设人均分到3个出口，每个出口最多承载总人数的1/3
        # 这样50人和1500人的密度值范围一致
        max_density_people = max(self.n_pedestrians / 3.0, 10.0)  # 最小基准10人
        density = min(len(nearby_peds) / max_density_people, 1.0)

        # 拥堵度 (基于平均速度下降) - 这个是相对值，不需要改
        if len(nearby_peds) > 0:
            avg_speed = np.mean([ped.speed for ped in nearby_peds])
            expected_speed = 1.2
            congestion = max(0, 1 - avg_speed / expected_speed)
        else:
            congestion = 0.0

        return density, congestion

    # ========== 预测性疏通系统 (Neural Network Enhanced) ==========

    def predict_future_positions(
        self,
        t_horizon: float = 5.0
    ) -> Dict[int, np.ndarray]:
        """预测所有行人t_horizon秒后的位置

        优先使用Social-LSTM神经网络预测，回退到线性外推

        Args:
            t_horizon: 预测时间范围（秒）

        Returns:
            字典 {行人ID: 预测位置}
        """
        future_positions = {}

        # 如果有神经网络预测器，使用完整轨迹预测
        if self.trajectory_predictor is not None:
            predictions = self.predict_future_positions_neural()
            # 取预测轨迹的最后一个点作为t_horizon后的位置
            for ped_id, traj in predictions.items():
                if len(traj) > 0:
                    future_positions[ped_id] = traj[-1]
            return future_positions

        # 回退到线性外推
        for ped in self.sfm.pedestrians:
            speed = np.linalg.norm(ped.velocity)

            if speed > 0.1:
                future_pos = ped.position + ped.velocity * t_horizon
            else:
                direction = ped.target - ped.position
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    future_pos = ped.position + (direction / dist) * ped.desired_speed * t_horizon
                else:
                    future_pos = ped.position.copy()

            # 限制在场景边界内
            future_pos[0] = np.clip(future_pos[0], 0, self.scene_width)
            future_pos[1] = np.clip(future_pos[1], 0, self.scene_height)

            future_positions[ped.id] = future_pos

        return future_positions

    def predict_future_positions_neural(self) -> Dict[int, np.ndarray]:
        """使用神经网络预测未来轨迹

        使用Social-LSTM模型预测每个行人未来12帧的轨迹

        Returns:
            字典 {行人ID: (12, 2) 预测轨迹}
        """
        if self.trajectory_predictor is None:
            return {}

        # 更新历史缓冲区
        for ped in self.sfm.pedestrians:
            self.trajectory_predictor.update_history(ped.id, ped.position)

        # 批量预测
        predictions = self.trajectory_predictor.predict_all_trajectories(
            self.sfm.pedestrians,
            scene_bounds=(0, 0, self.scene_width, self.scene_height)
        )

        return predictions

    def detect_corner_trap(
        self,
        ped_id: int,
        pred_trajectory: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """检测行人是否正在走向死角

        Args:
            ped_id: 行人ID
            pred_trajectory: 预测轨迹 (pred_len, 2)

        Returns:
            is_trapped: 是否将陷入角落
            trap_corner: 陷阱角落位置
        """
        if self.trajectory_predictor is not None:
            return self.trajectory_predictor.detect_corner_trap(
                pred_trajectory,
                self.corner_traps,
                trap_radius=3.0
            )

        # 简单的角落检测 (回退方案)
        for future_pos in pred_trajectory:
            for corner in self.corner_traps:
                if np.linalg.norm(future_pos - corner) < 3.0:
                    return True, corner
        return False, None

    def proactive_corner_avoidance(self) -> int:
        """主动角落避免: 检测并重定向将陷入角落的行人

        Returns:
            重定向的行人数量
        """
        if self.trajectory_predictor is None:
            return 0

        predictions = self.predict_future_positions_neural()
        redirected_count = 0

        for ped in self.sfm.pedestrians:
            if ped.id not in predictions:
                continue

            pred_traj = predictions[ped.id]
            is_trapped, trap_corner = self.detect_corner_trap(ped.id, pred_traj)

            if is_trapped:
                # 找到最近的非陷阱出口
                current_dist = np.linalg.norm(ped.position - ped.target)
                best_exit = None
                best_dist = float('inf')

                for exit_obj in self.exits:
                    exit_pos = exit_obj.position
                    # 检查这个出口是否远离陷阱
                    dist_to_trap = np.linalg.norm(exit_pos - trap_corner)
                    if dist_to_trap > 5.0:  # 出口距离陷阱足够远
                        dist = np.linalg.norm(ped.position - exit_pos)
                        if dist < best_dist:
                            best_dist = dist
                            best_exit = exit_obj

                # 重定向到安全出口
                if best_exit is not None:
                    ped.target = best_exit.position.copy()
                    redirected_count += 1

        return redirected_count

    def predict_exit_congestion(
        self,
        future_positions: Dict[int, np.ndarray],
        detection_radius: float = 10.0
    ) -> Dict[str, int]:
        """预测每个出口的拥堵程度

        统计预测位置在每个出口检测范围内的行人数量

        Args:
            future_positions: 预测的行人位置 {行人ID: 位置}
            detection_radius: 出口检测半径

        Returns:
            字典 {出口名称: 预测人数}
        """
        exit_counts = {exit_obj.name: 0 for exit_obj in self.exits}

        # 建立行人ID到行人对象的映射
        ped_map = {ped.id: ped for ped in self.sfm.pedestrians}

        for ped_id, future_pos in future_positions.items():
            ped = ped_map.get(ped_id)
            if ped is None:
                continue

            # 根据行人当前目标确定其去向的出口
            min_dist = float('inf')
            target_exit_name = None

            for exit_obj in self.exits:
                dist_to_exit = np.linalg.norm(ped.target - exit_obj.position)
                if dist_to_exit < min_dist:
                    min_dist = dist_to_exit
                    target_exit_name = exit_obj.name

            # 如果行人的预测位置在目标出口附近，计入该出口
            if target_exit_name:
                for exit_obj in self.exits:
                    if exit_obj.name == target_exit_name:
                        dist_to_exit = np.linalg.norm(future_pos - exit_obj.position)
                        if dist_to_exit < detection_radius:
                            exit_counts[target_exit_name] += 1
                        break

        return exit_counts

    def _compute_redirect_cost(
        self,
        ped: 'Pedestrian',
        current_exit: Exit,
        new_exit: Exit
    ) -> float:
        """计算行人改道的成本

        成本函数:
        cost = distance_to_new_exit - distance_to_current_exit
               + 0.5 * congestion_at_new_exit

        Args:
            ped: 行人对象
            current_exit: 当前目标出口
            new_exit: 新目标出口

        Returns:
            改道成本（越小越适合改道）
        """
        # 距离成本
        dist_current = np.linalg.norm(ped.position - current_exit.position)
        dist_new = np.linalg.norm(ped.position - new_exit.position)
        distance_cost = dist_new - dist_current

        # 拥堵成本
        _, congestion_new = self._compute_exit_metrics(new_exit)
        congestion_cost = 0.5 * congestion_new * 10  # 归一化

        return distance_cost + congestion_cost

    def rebalance_exit_assignments(
        self,
        threshold: int = None,
        t_horizon: float = 5.0
    ) -> int:
        """重新分配行人到出口，预防性避免拥堵

        核心算法:
        1. 预测每个出口未来的人数
        2. 找出预测过载的出口
        3. 选择改道成本最小的行人重新分配

        Args:
            threshold: 单出口人数阈值，超过则视为过载
                       如果为None，自动计算为 n_pedestrians/3 * 1.5
            t_horizon: 预测时间范围

        Returns:
            重新分配的行人数量
        """
        # 动态阈值：每出口平均承载的1.5倍视为过载
        if threshold is None:
            threshold = max(int(self.n_pedestrians / 3 * 1.5), 5)
        # 1. 预测未来位置
        future_positions = self.predict_future_positions(t_horizon)

        # 2. 预测每个出口的人数
        exit_counts = self.predict_exit_congestion(future_positions)

        # 3. 找出过载的出口
        overloaded_exits = [
            exit_obj for exit_obj in self.exits
            if exit_counts[exit_obj.name] > threshold
        ]

        if not overloaded_exits:
            return 0

        redirect_count = 0
        ped_map = {ped.id: ped for ped in self.sfm.pedestrians}

        # 4. 对每个过载出口，重分配部分行人
        for overloaded_exit in overloaded_exits:
            excess = exit_counts[overloaded_exit.name] - threshold

            # 找出目标是该出口的行人
            candidates = []
            for ped in self.sfm.pedestrians:
                # 检查行人的目标是否是过载出口
                dist_to_overloaded = np.linalg.norm(ped.target - overloaded_exit.position)
                if dist_to_overloaded < 1.0:  # 目标是该出口
                    # 只考虑还未太靠近出口的行人（有重定向空间）
                    current_dist = np.linalg.norm(ped.position - overloaded_exit.position)
                    if current_dist > 8.0:  # 距离出口还有一定距离
                        candidates.append(ped)

            if not candidates:
                continue

            # 找出可替代的出口（非过载）
            alternative_exits = [
                exit_obj for exit_obj in self.exits
                if exit_obj.id != overloaded_exit.id and
                exit_counts[exit_obj.name] < threshold
            ]

            if not alternative_exits:
                continue

            # 计算每个候选行人的改道成本
            redirect_options = []
            for ped in candidates:
                for alt_exit in alternative_exits:
                    cost = self._compute_redirect_cost(ped, overloaded_exit, alt_exit)
                    redirect_options.append((ped, alt_exit, cost))

            # 按成本排序，选择成本最小的行人改道
            redirect_options.sort(key=lambda x: x[2])

            # 重定向excess个行人（但不超过候选人数）
            redirected_peds = set()
            for ped, alt_exit, cost in redirect_options:
                if len(redirected_peds) >= excess:
                    break
                if ped.id in redirected_peds:
                    continue
                # 只有成本不太高时才改道（避免绕太远的路）
                if cost < 15.0:
                    ped.target = alt_exit.position.copy()
                    redirected_peds.add(ped.id)
                    redirect_count += 1

        return redirect_count

    # ========== 分层预测式引导系统 ==========

    def predictive_guidance_system(self) -> int:
        """分层预测式引导系统 - 替代原有的随机概率引导

        第1层：PPO全局决策 - 获取推荐出口
        第2层：Social-LSTM预测筛选 - 识别将遇到问题的行人
        第3层：个体决策 - 检查引导条件后引导

        Returns:
            本次引导的行人数量
        """
        if self.trajectory_predictor is None:
            return 0

        guided_count = 0
        current_time = self.current_step * self.dt

        # 第1层：PPO推荐的出口（由step()传入的last_action决定）
        recommended_exit = self.exits[self.last_action]

        # 第2层：预测所有人轨迹，识别问题行人
        predictions = self.predict_future_positions_neural()
        problem_pedestrians = self._identify_problem_pedestrians(predictions)

        # 第3层：对问题行人进行个体引导决策
        for ped in problem_pedestrians:
            if self._can_be_guided(ped, current_time):
                best_exit = self._find_best_alternative_exit(ped, recommended_exit)
                if best_exit is not None:
                    self._apply_guidance(ped, best_exit, current_time)
                    guided_count += 1

        return guided_count

    def _identify_problem_pedestrians(
        self,
        predictions: Dict[int, np.ndarray]
    ) -> List[Pedestrian]:
        """识别将遇到问题的行人（主动预防式）

        问题类型（优先级从高到低）：
        1. 走向角落陷阱
        2. 走向已拥堵的出口（实时拥堵）
        3. 走向负载过高的出口（负载不均衡）

        关键：只从过载出口分流行人，不会把人分流到过载出口

        Args:
            predictions: 神经网络预测的轨迹 {ped_id: (12, 2)}

        Returns:
            问题行人列表
        """
        problem_peds = []
        cfg = GUIDANCE_CONFIG

        # 第一步：统计每个出口的目标人数
        exit_target_count = {exit_obj.id: 0 for exit_obj in self.exits}

        for ped in self.sfm.pedestrians:
            for exit_obj in self.exits:
                if np.linalg.norm(ped.target - exit_obj.position) < 2.0:
                    exit_target_count[exit_obj.id] += 1
                    break

        # 计算总人数
        total_peds = len(self.sfm.pedestrians)
        if total_peds < cfg['min_peds_for_rebalance']:
            return problem_peds

        # 第二步：找出过载的出口（目标人数占比>阈值）
        avg_per_exit = total_peds / len(self.exits)
        overloaded_exit_ids = set()

        for exit_obj in self.exits:
            count = exit_target_count[exit_obj.id]
            ratio = count / total_peds if total_peds > 0 else 0

            # 过载条件：占比超过阈值 且 人数明显高于平均
            if ratio > cfg['exit_imbalance_threshold'] and count > avg_per_exit * 1.2:
                overloaded_exit_ids.add(exit_obj.id)

        # 第三步：识别问题行人（只从过载出口选择）
        for ped in self.sfm.pedestrians:
            if ped.id not in predictions:
                continue

            pred_traj = predictions[ped.id]

            # 检查1：走向角落陷阱
            is_trapped, _ = self.detect_corner_trap(ped.id, pred_traj)
            if is_trapped:
                problem_peds.append(ped)
                continue

            # 检查2：走向已拥堵的出口（实时拥堵度）
            if self._will_reach_congested_exit(ped, pred_traj):
                problem_peds.append(ped)
                continue

            # 检查3：走向过载出口的行人（负载分流）
            # 关键：只有目标是过载出口的行人才会被选中分流
            for exit_obj in self.exits:
                if exit_obj.id in overloaded_exit_ids:
                    if np.linalg.norm(ped.target - exit_obj.position) < 2.0:
                        # 该行人正走向过载出口
                        dist_to_exit = np.linalg.norm(ped.position - exit_obj.position)
                        rebalance_dist = cfg.get('rebalance_distance_threshold', 10.0)
                        # 只分流距离出口较远的行人（有调整空间）
                        if dist_to_exit > rebalance_dist:
                            problem_peds.append(ped)
                        break

        return problem_peds

    def _will_reach_congested_exit(
        self,
        ped: Pedestrian,
        pred_traj: np.ndarray
    ) -> bool:
        """检查行人是否正在走向拥堵出口

        Args:
            ped: 行人对象
            pred_traj: 预测轨迹

        Returns:
            是否将到达拥堵出口
        """
        cfg = GUIDANCE_CONFIG

        # 找到行人当前目标对应的出口
        target_exit = None
        min_dist = float('inf')
        for exit_obj in self.exits:
            dist = np.linalg.norm(ped.target - exit_obj.position)
            if dist < min_dist:
                min_dist = dist
                target_exit = exit_obj

        if target_exit is None:
            return False

        # 计算该出口的拥堵度
        _, congestion = self._compute_exit_metrics(target_exit)

        # 如果拥堵度超过阈值，判定为问题
        return congestion > cfg['congestion_threshold']

    def _can_be_guided(self, ped: Pedestrian, current_time: float) -> bool:
        """检查行人是否可以被引导

        条件：
        1. 在引导区域内（已过闸机）
        2. 引导次数未超限
        3. 冷却时间已过
        4. 距离目标足够远

        Args:
            ped: 行人对象
            current_time: 当前仿真时间

        Returns:
            是否可以被引导
        """
        cfg = GUIDANCE_CONFIG

        # 条件1：在引导区域内（x > guidance_zone_x，即已过闸机）
        if ped.position[0] <= cfg['guidance_zone_x']:
            return False

        # 条件2：引导次数未超限
        if ped.guidance_count >= cfg['max_guidance_count']:
            return False

        # 条件3：冷却时间已过
        if current_time - ped.last_guidance_time < cfg['cooldown_time']:
            return False

        # 条件4：距离目标足够远（有改道空间）
        dist_to_target = np.linalg.norm(ped.position - ped.target)
        if dist_to_target < cfg['min_distance_to_target']:
            return False

        return True

    def _find_best_alternative_exit(
        self,
        ped: Pedestrian,
        recommended_exit: 'Exit'
    ) -> Optional['Exit']:
        """为问题行人找到最佳替代出口

        核心原则：
        1. 不引导到过载出口（目标人数过多的出口）
        2. 优先引导到人少且近的出口
        3. 只有明显更优时才改道

        Args:
            ped: 行人对象
            recommended_exit: PPO推荐的出口

        Returns:
            最佳替代出口，如果没有则返回None
        """
        cfg = GUIDANCE_CONFIG

        # 当前目标出口
        current_target_exit = None
        current_dist = float('inf')
        for exit_obj in self.exits:
            if np.linalg.norm(ped.target - exit_obj.position) < 1.0:
                current_target_exit = exit_obj
                current_dist = np.linalg.norm(ped.position - exit_obj.position)
                break

        if current_target_exit is None:
            return None

        # 统计各出口目标人数，确定过载出口
        total_peds = len(self.sfm.pedestrians)
        exit_target_count = {exit_obj.id: 0 for exit_obj in self.exits}
        for p in self.sfm.pedestrians:
            for exit_obj in self.exits:
                if np.linalg.norm(p.target - exit_obj.position) < 2.0:
                    exit_target_count[exit_obj.id] += 1
                    break

        # 计算当前出口的拥堵度
        _, current_congestion = self._compute_exit_metrics(current_target_exit)
        current_count = exit_target_count[current_target_exit.id]

        # 寻找最佳替代出口
        best_exit = None
        best_score = float('-inf')

        for exit_obj in self.exits:
            if exit_obj.id == current_target_exit.id:
                continue

            # 关键检查：不引导到人更多的出口
            target_count = exit_target_count[exit_obj.id]
            if target_count >= current_count:
                continue  # 新出口人不比当前少，不考虑

            _, congestion = self._compute_exit_metrics(exit_obj)
            dist_to_exit = np.linalg.norm(ped.position - exit_obj.position)

            # 评分：人数差距 + 距离因素
            # 人数差距更重要（权重更高）
            count_benefit = (current_count - target_count) * 5  # 每少1人 = +5分
            distance_cost = (dist_to_exit - current_dist) * 0.5  # 每远1米 = -0.5分
            congestion_benefit = (current_congestion - congestion) * 10

            score = count_benefit + congestion_benefit - distance_cost

            # 只有分数为正（确实更好）才考虑
            if score > 0 and score > best_score:
                best_score = score
                best_exit = exit_obj

        return best_exit

    def _apply_guidance(
        self,
        ped: Pedestrian,
        new_exit: 'Exit',
        current_time: float
    ) -> None:
        """应用引导并更新行人状态

        Args:
            ped: 行人对象
            new_exit: 新目标出口
            current_time: 当前仿真时间
        """
        # 记录原始目标（首次引导时）
        if ped.original_target is None:
            ped.original_target = ped.target.copy()

        old_target = ped.target.copy()

        # 更新目标
        ped.target = new_exit.position.copy()

        # 更新引导状态
        ped.guidance_count += 1
        ped.last_guidance_time = current_time

        # 调试日志（可选）
        # print(f"引导行人 {ped.id}: 第{ped.guidance_count}次, 目标变更为出口{new_exit.name}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步

        Args:
            action: 选择的主要引导出口 ID (0=A, 1=B, 2=C)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 保存PPO决策，供分层引导系统使用
        self.last_action = action

        # ========== 分层预测式引导系统 ==========
        # 替代原有的随机概率引导(_apply_action)
        # 特点：定向引导、一次决策、有冷却期、基于预测
        guided_count = 0
        corner_avoided = 0

        if self.current_step % 5 == 0 and len(self.sfm.pedestrians) > 5:
            # 使用分层预测式引导系统
            if self.trajectory_predictor is not None:
                guided_count = self.predictive_guidance_system()
                # 主动角落避免（作为补充）
                corner_avoided = self.proactive_corner_avoidance()
            else:
                # 回退到旧版负载均衡（无神经网络预测时）
                # 动态阈值：每出口平均承载的1.5倍视为过载
                dynamic_threshold = max(int(self.n_pedestrians / 3 * 1.5), 5)
                guided_count = self.rebalance_exit_assignments(
                    threshold=dynamic_threshold,
                    t_horizon=3.0
                )

        # 运行社会力模型多步
        for _ in range(5):
            self.sfm.step(self.dt)

        # 大规模场景：恐慌传播 (每10步检测一次)
        panic_spread_count = 0
        if self.current_step % 10 == 0 and self.n_pedestrians > 100:
            panic_spread_count = self._spread_panic()

        # 检查并移除已到达出口的行人
        self._check_evacuated()

        self.current_step += 1

        # 计算奖励
        reward = self._compute_reward()
        self.history['rewards'].append(reward)

        # 检查终止条件
        terminated = len(self.sfm.pedestrians) == 0
        truncated = self.current_step >= self.max_steps

        # 大规模安全检测
        max_density, danger_count = self._detect_crush_risk()

        obs = self._get_observation()
        info = {
            'evacuated': self.evacuated_count,
            'remaining': len(self.sfm.pedestrians),
            'step': self.current_step,
            'evacuated_by_exit': self.evacuated_by_exit.copy(),
            'evacuated_by_type': {t.value: c for t, c in self.evacuated_by_type.items()},
            'enhanced_behaviors': self.enable_enhanced_behaviors,
            'guided_this_step': guided_count,  # 本步引导的行人数（分层预测式引导）
            'corner_avoided_this_step': corner_avoided,  # 本步角落避免的行人数
            'neural_prediction': self.trajectory_predictor is not None,  # 是否使用神经网络预测
            # 大规模安全指标
            'max_local_density': max_density,        # 最大局部密度 (人/m²)
            'danger_zone_count': danger_count,       # 危险区域行人数
            'panic_spread_count': panic_spread_count,  # 本步新增恐慌人数
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

        增强版本: 同时统计按类型的疏散数量，并清理轨迹预测历史
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
            # 清理轨迹预测历史
            if self.trajectory_predictor is not None:
                self.trajectory_predictor.remove_pedestrian(ped.id)

        self.history['evacuated'].append(self.evacuated_count)

    def _compute_reward(self) -> float:
        """计算增强奖励 (归一化版本)

        奖励组成:
        1. 疏散奖励: 疏散比例的奖励 (归一化)
        2. 拥堵惩罚: 出口拥堵的惩罚
        3. 时间惩罚: 每步的时间成本
        4. 完成奖励: 全部疏散的奖励
        5. 均衡奖励: 出口分布均匀的奖励 (归一化)
        6. 人流效率奖励: 疏散速率提升的奖励 (归一化)
        7. 安全间距奖励: 保持安全距离的奖励

        归一化设计: 所有奖励信号相对于人数比例计算，使模型能从小规模泛化到大规模
        """
        reward = 0.0
        w = self.reward_weights

        # 1. 疏散奖励：按疏散比例计算，而非绝对人数
        # 这样50人疏散5人(10%) 和 1500人疏散150人(10%) 得到相同奖励
        new_evacuated = self.evacuated_count - self.last_evacuated_count
        evacuation_ratio = new_evacuated / max(self.n_pedestrians, 1)
        reward += evacuation_ratio * w["evac_per_person"] * 100  # 乘100使数值合理

        # 更新疏散速率缓冲区 (存储比例而非绝对数)
        self.evacuation_rate_buffer.pop(0)
        self.evacuation_rate_buffer.append(evacuation_ratio)
        self.last_evacuated_count = self.evacuated_count

        # 2. 拥堵惩罚 (已经是归一化的0-1值)
        total_congestion = 0
        for exit_obj in self.exits:
            _, congestion = self._compute_exit_metrics(exit_obj)
            total_congestion += congestion
        reward -= total_congestion * w["congestion_penalty"]
        self.history['congestion'].append(total_congestion)

        # 3. 时间惩罚（鼓励快速疏散）
        reward -= w["time_penalty"]

        # 4. 完成奖励
        if len(self.sfm.pedestrians) == 0:
            reward += w["completion_bonus"]
            # 额外奖励：快速完成
            time_bonus = max(0, (self.max_steps - self.current_step) / self.max_steps * 50)
            reward += time_bonus

        # 5. 均衡奖励：鼓励各出口分流 (归一化)
        counts = list(self.evacuated_by_exit.values())
        total_evacuated = sum(counts)
        if total_evacuated > 0:
            # 计算分布均匀度（用变异系数而非方差，自动归一化）
            mean_count = total_evacuated / 3
            if mean_count > 0:
                std_count = np.sqrt(sum((c - mean_count) ** 2 for c in counts) / 3)
                cv = std_count / mean_count  # 变异系数 (0-1范围)
                balance_penalty = min(cv, 1.0)
            else:
                balance_penalty = 0.0
            reward -= balance_penalty * w["balance_penalty"]

        # 6. 人流效率奖励：疏散速率提升 (已归一化)
        if len(self.evacuation_rate_buffer) >= 2:
            rate_improvement = self.evacuation_rate_buffer[-1] - self.evacuation_rate_buffer[-2]
            if rate_improvement > 0:
                reward += rate_improvement * w.get("evacuation_rate_bonus", 2.0) * 100

        # 7. 安全间距奖励：避免过度拥挤 (已经是0-1)
        safety_reward = self._compute_safety_distance_reward()
        reward += safety_reward * w.get("safety_distance_bonus", 0.5)

        # 8. 踩踏风险惩罚 (大规模安全机制)
        max_density, danger_count = self._detect_crush_risk()
        if danger_count > 0:
            # 危险区域有人时给予惩罚
            crush_penalty = (danger_count / max(self.n_pedestrians, 1)) * w.get("crush_penalty", 10.0)
            reward -= crush_penalty

        return reward

    def _compute_safety_distance_reward(self) -> float:
        """计算安全间距奖励

        基于行人之间的平均距离，鼓励保持安全距离

        Returns:
            安全间距奖励 (0-1, 越大越好)
        """
        if len(self.sfm.pedestrians) < 2:
            return 1.0

        # 计算平均最近邻距离
        min_distances = []
        positions = [ped.position for ped in self.sfm.pedestrians]

        for i, pos_i in enumerate(positions):
            min_dist = float('inf')
            for j, pos_j in enumerate(positions):
                if i != j:
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < min_dist:
                        min_dist = dist
            if min_dist < float('inf'):
                min_distances.append(min_dist)

        if not min_distances:
            return 1.0

        avg_min_dist = np.mean(min_distances)

        # 安全距离阈值: 0.8米 (基于人体宽度)
        safe_distance = 0.8

        if avg_min_dist >= safe_distance:
            return 1.0
        else:
            # 距离越近，奖励越低
            return max(0, avg_min_dist / safe_distance)

    def _detect_crush_risk(self) -> Tuple[float, int]:
        """检测踩踏风险 (大规模场景安全机制)

        基于局部密度检测危险区域:
        - 密度 > 4人/m² : 极危险，可能踩踏
        - 密度 > 2.5人/m² : 警告，需要分流

        Returns:
            (最大局部密度, 危险区域行人数)
        """
        if len(self.sfm.pedestrians) < 10:
            return 0.0, 0

        safety = LARGE_SCALE_SAFETY
        critical_density = safety["critical_density"]
        warning_density = safety["warning_density"]

        max_density = 0.0
        danger_count = 0
        detection_radius = 2.0  # 2米半径检测局部密度

        for ped in self.sfm.pedestrians:
            # 计算该行人周围的局部密度
            nearby_count = 0
            for other in self.sfm.pedestrians:
                if other.id != ped.id:
                    dist = np.linalg.norm(ped.position - other.position)
                    if dist < detection_radius:
                        nearby_count += 1

            # 局部密度 = 人数 / 圆面积
            local_density = nearby_count / (np.pi * detection_radius ** 2)
            max_density = max(max_density, local_density)

            if local_density > critical_density:
                danger_count += 1

        return max_density, danger_count

    def _spread_panic(self) -> int:
        """恐慌传播机制 (大规模场景)

        当一个行人恐慌时，周围的人也会受到影响
        传播规则:
        - 恐慌因子 > 0.3 的行人会传播恐慌
        - 传播半径内的行人恐慌因子增加
        - 传播强度随距离衰减

        Returns:
            新增恐慌的行人数
        """
        if not hasattr(self.sfm, 'enable_panic') or not self.sfm.enable_panic:
            return 0

        safety = LARGE_SCALE_SAFETY
        spread_radius = safety["panic_spread_radius"]
        spread_rate = safety["panic_spread_rate"]

        new_panic_count = 0

        # 找出当前恐慌的行人
        panicked_peds = [ped for ped in self.sfm.pedestrians
                        if hasattr(ped, 'panic_factor') and ped.panic_factor > 0.3]

        if not panicked_peds:
            return 0

        # 传播恐慌
        for source in panicked_peds:
            for target in self.sfm.pedestrians:
                if target.id == source.id:
                    continue

                dist = np.linalg.norm(target.position - source.position)
                if dist < spread_radius and dist > 0:
                    # 传播强度随距离衰减
                    spread_strength = spread_rate * (1 - dist / spread_radius)
                    spread_strength *= source.panic_factor  # 源头恐慌程度也影响传播

                    old_panic = target.panic_factor if hasattr(target, 'panic_factor') else 0
                    target.panic_factor = min(0.5, old_panic + spread_strength)

                    if old_panic < 0.1 and target.panic_factor >= 0.1:
                        new_panic_count += 1

        return new_panic_count

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
