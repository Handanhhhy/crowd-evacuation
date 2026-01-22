"""
成都东客站地铁出站口疏散仿真 - 集成PPO智能引导
社会力模型 + PPO强化学习 + GBM行为预测 + Social-LSTM轨迹预测

增强版本:
- 多种行人类型可视化（不同颜色）
- GBM行为预测器状态显示
- 行人行为状态显示（等待、恐慌）
- Social-LSTM神经网络轨迹预测可视化

注意: Pygame显示使用英文避免字体编码问题
"""

# 在Apple Silicon上，必须在导入torch之前设置环境变量以避免MPS相关的segfault
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

# 在导入任何依赖torch的模块之前，强制使用CPU
import torch
torch.set_default_device('cpu')

import pygame
from stable_baselines3 import PPO
from sfm.social_force import (
    SocialForceModel,
    Pedestrian,
    PedestrianType,
    PEDESTRIAN_TYPE_PARAMS
)

# 尝试导入轨迹预测器
try:
    from ml.trajectory_predictor import TrajectoryPredictor
    TRAJECTORY_PREDICTOR_AVAILABLE = True
except ImportError:
    TRAJECTORY_PREDICTOR_AVAILABLE = False


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


# 颜色定义
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
BLUE = (70, 130, 180)
RED = (220, 80, 80)
ORANGE = (255, 165, 0)
GREEN = (60, 179, 113)
LIGHT_GREEN = (144, 238, 144)
DARK_BLUE = (25, 25, 112)
YELLOW = (255, 215, 0)
PURPLE = (147, 112, 219)
CYAN = (0, 255, 255)
PINK = (255, 182, 193)
BG_COLOR = (240, 240, 235)
FLOOR_COLOR = (220, 220, 210)
GATE_COLOR = (100, 100, 100)
PREDICTION_COLOR = (100, 200, 255)  # 预测轨迹颜色
CORNER_TRAP_COLOR = (255, 100, 100)  # 角落陷阱颜色

# 行人类型颜色映射 (基于文献参数配置)
PEDESTRIAN_TYPE_COLORS = {
    PedestrianType.NORMAL: BLUE,        # 普通成年人: 蓝色
    PedestrianType.ELDERLY: GREEN,      # 老人: 绿色
    PedestrianType.CHILD: YELLOW,       # 儿童: 黄色
    PedestrianType.IMPATIENT: RED,      # 急躁型: 红色
}


class MetroStationWithPPO:
    """成都东客站地铁出站口 - PPO智能引导版"""

    def __init__(
        self,
        n_pedestrians: int = 80,
        scale: float = 12.0,
        dt: float = 0.05,
        use_ppo: bool = True,
        enable_enhanced_behaviors: bool = True,
        show_type_colors: bool = True,
        type_distribution: dict = None,
        show_predictions: bool = True,
        enable_neural_prediction: bool = True
    ):
        """
        Args:
            n_pedestrians: 行人数量
            scale: 像素/米比例
            dt: 时间步长
            use_ppo: 是否使用PPO智能引导
            enable_enhanced_behaviors: 是否启用增强行为 (等待、犹豫、恐慌)
            show_type_colors: 是否按类型显示颜色 (否则按速度显示)
            type_distribution: 行人类型分布比例
            show_predictions: 是否显示预测轨迹
            enable_neural_prediction: 是否启用神经网络轨迹预测
        """
        self.n_pedestrians = n_pedestrians
        self.scale = scale
        self.dt = dt
        self.use_ppo = use_ppo
        self.enable_enhanced_behaviors = enable_enhanced_behaviors
        self.show_type_colors = show_type_colors
        self.show_predictions = show_predictions
        self.enable_neural_prediction = enable_neural_prediction

        # 行人类型分布 (默认: 70%普通 + 15%老人 + 10%儿童 + 5%急躁)
        self.type_distribution = type_distribution or {
            PedestrianType.NORMAL: 0.70,
            PedestrianType.ELDERLY: 0.15,
            PedestrianType.CHILD: 0.10,
            PedestrianType.IMPATIENT: 0.05,
        }

        # 场景尺寸
        self.scene_width = 60.0
        self.scene_height = 40.0

        # 窗口大小 (增加信息面板宽度)
        self.window_width = int(self.scene_width * scale) + 260
        self.window_height = int(self.scene_height * scale) + 60

        # 闸机位置
        self.gates = []
        gate_y_positions = [10, 15, 20, 25, 30]
        for y in gate_y_positions:
            self.gates.append({
                'position': np.array([20.0, y]),
                'width': 0.8,
                'length': 3.0
            })

        # Exit definitions
        self.exits = [
            {'id': 0, 'name': 'A', 'position': np.array([60, 10]), 'width': 4.0, 'label': 'Exit A'},
            {'id': 1, 'name': 'B', 'position': np.array([60, 30]), 'width': 4.0, 'label': 'Exit B'},
            {'id': 2, 'name': 'C', 'position': np.array([40, 40]), 'width': 5.0, 'label': 'Exit C (Main)'},
        ]

        # 柱子位置
        self.pillars = [
            np.array([30, 12]), np.array([30, 28]),
            np.array([45, 12]), np.array([45, 28]),
            np.array([35, 20]), np.array([50, 20]),
        ]

        # Facilities
        self.facilities = [
            {'type': 'info', 'position': np.array([32, 20]), 'size': (3, 2), 'label': 'Info'},
            {'type': 'stairs', 'position': np.array([55, 20]), 'size': (4, 8), 'label': 'Stairs'},
        ]

        # PPO模型
        self.ppo_model = None
        self.current_action = 0  # 当前PPO推荐的出口
        self.ppo_update_interval = 10  # 每10步更新一次PPO决策
        self.is_metro_model = False  # 是否使用地铁站专用模型（3出口）

        # GBM行为预测模型
        self.gbm_predictor = None
        self.gbm_loaded = False

        # Social-LSTM轨迹预测模型
        self.trajectory_predictor = None
        self.trajectory_loaded = False
        self.predicted_trajectories = {}  # {ped_id: (pred_len, 2)}
        self.corner_avoided_count = 0

        # 角落陷阱位置
        self.corner_traps = [
            np.array([60, 40]),   # Exit C 上方右角落
            np.array([60, 0]),    # 右下角落
            np.array([55, 16]),   # 楼梯旁边
            np.array([55, 24]),   # 楼梯旁边
        ]

        # 分层预测式引导系统统计
        self.total_guided_count = 0        # 总引导次数
        self.current_step_guided = 0       # 本步引导人数
        self.guidance_stats = {            # 引导统计
            'by_reason': {'congestion': 0, 'corner_trap': 0},
            'by_exit': {'A': 0, 'B': 0, 'C': 0}
        }

        # 状态
        self.model = None
        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.evacuated_by_type = {t: 0 for t in PedestrianType}
        self.step_count = 0
        self.running = True
        self.paused = False

        # 对比模式
        self.show_comparison = False

    def load_ppo_model(self):
        """加载训练好的PPO模型"""
        # 优先加载地铁站专用模型
        metro_model_path = project_root / "outputs" / "models" / "ppo_metro.zip"
        fallback_model_path = project_root / "outputs" / "models" / "ppo_evacuation.zip"

        if metro_model_path.exists():
            model_path = metro_model_path
            self.is_metro_model = True
        elif fallback_model_path.exists():
            model_path = fallback_model_path
            self.is_metro_model = False
        else:
            model_path = None

        if model_path is not None and self.use_ppo:
            try:
                self.ppo_model = PPO.load(str(model_path))
                obs_dim = self.ppo_model.observation_space.shape[0]
                print(f"PPO模型已加载: {model_path}")
                if obs_dim >= 16:
                    print(f"  (增强版16维观测模型)")
                elif obs_dim == 8:
                    print(f"  (地铁站专用8维观测模型)")
                else:
                    print(f"  (旧版{obs_dim}维观测模型)")
                print(f"  提示: 如需训练16维增强版，运行 python examples/train_ppo_metro.py")
                return True
            except Exception as e:
                print(f"加载PPO模型失败: {e}")
                self.ppo_model = None
                return False
        else:
            print("未找到PPO模型或未启用PPO")
            print("  提示: 运行 python examples/train_ppo_metro.py 训练地铁站模型")
            return False

    def load_gbm_predictor(self):
        """加载GBM行为预测器（基于ETH/UCY真实数据训练）"""
        gbm_model_path = project_root / "outputs" / "models" / "gbm_behavior.joblib"

        if gbm_model_path.exists():
            try:
                from ml.gbm_predictor import GBMPredictor
                self.gbm_predictor = GBMPredictor()
                self.gbm_predictor.load(str(gbm_model_path))
                self.gbm_loaded = True
                print(f"GBM Behavior Predictor loaded: {gbm_model_path}")
                print("  (Trained on ETH/UCY real pedestrian trajectory data)")
                return True
            except Exception as e:
                print(f"Failed to load GBM model: {e}")
                self.gbm_predictor = None
                self.gbm_loaded = False
                return False
        else:
            print("GBM Behavior Predictor not found")
            print("  Hint: Run 'python examples/train_gbm_behavior.py' to train")
            self.gbm_predictor = None
            self.gbm_loaded = False
            return False

    def load_trajectory_predictor(self):
        """加载轨迹预测器 (支持Social-LSTM和Trajectron++)"""
        if not TRAJECTORY_PREDICTOR_AVAILABLE:
            print("Trajectory Predictor module not available")
            return False

        if not self.enable_neural_prediction:
            print("Neural trajectory prediction disabled")
            return False

        # 优先加载Trajectron++，回退到Social-LSTM
        trajectron_path = project_root / "outputs" / "models" / "trajectron.pt"
        lstm_model_path = project_root / "outputs" / "models" / "social_lstm.pt"

        # 选择模型路径
        if trajectron_path.exists():
            model_path = str(trajectron_path)
            model_type = 'trajectron'
        elif lstm_model_path.exists():
            model_path = str(lstm_model_path)
            model_type = 'social_lstm'
        else:
            model_path = None
            model_type = 'auto'

        try:
            self.trajectory_predictor = TrajectoryPredictor(
                model_path=model_path,
                obs_len=8,
                pred_len=12,
                device='cpu',
                model_type=model_type
            )
            # 立即触发模型加载
            self.trajectory_predictor._ensure_model_loaded()
            self.trajectory_loaded = True

            if self.trajectory_predictor.use_neural_network:
                actual_type = self.trajectory_predictor.actual_model_type
                if actual_type == 'trajectron':
                    print(f"Trajectron++ Trajectory Predictor loaded: {trajectron_path}")
                    print("  Mode: Multi-modal GNN (Trajectron++)")
                else:
                    print(f"Social-LSTM Trajectory Predictor loaded: {lstm_model_path}")
                    print("  Mode: Neural Network (Social-LSTM)")
            else:
                print("Trajectory model not found, using linear extrapolation")
                print("  Hint: Run 'python examples/train_trajectron.py' (recommended)")
                print("        or 'python examples/train_trajectory.py' (Social-LSTM)")
            return True
        except Exception as e:
            print(f"Failed to load Trajectory Predictor: {e}")
            self.trajectory_predictor = None
            self.trajectory_loaded = False
            return False

    def world_to_screen(self, pos):
        """世界坐标转屏幕坐标"""
        x = int(pos[0] * self.scale) + 50
        y = int((self.scene_height - pos[1]) * self.scale) + 30
        return (x, y)

    def setup_model(self):
        """初始化社会力模型

        增强版本:
        - 集成GBM行为预测（从ETH/UCY真实数据学习）
        - 支持等待、犹豫、恐慌等行为
        - 支持多种行人类型（老人、儿童、急躁型）
        """
        # 创建集成GBM的增强版社会力模型
        self.model = SocialForceModel(
            tau=0.5,
            A=2000.0,
            B=0.08,
            wall_A=5000.0,    # 增强障碍物排斥力，防止卡住
            wall_B=0.1,       # 增大检测范围，更早发现障碍物
            # 增强行为参数
            enable_waiting=self.enable_enhanced_behaviors,
            enable_perturbation=self.enable_enhanced_behaviors,
            enable_panic=self.enable_enhanced_behaviors,
            waiting_density_threshold=0.8,
            perturbation_sigma=0.05,  # 减小随机扰动，GBM处理行为
            panic_density_threshold=1.5,
            # GBM行为预测器（基于ETH/UCY真实数据训练）
            gbm_predictor=self.gbm_predictor,
            gbm_weight=0.3,   # 融合权重: 30% GBM + 70% SFM
        )

        # 外墙
        self.model.add_obstacle(np.array([0, self.scene_height]), np.array([35, self.scene_height]))
        self.model.add_obstacle(np.array([45, self.scene_height]), np.array([self.scene_width, self.scene_height]))
        self.model.add_obstacle(np.array([0, 0]), np.array([self.scene_width, 0]))
        self.model.add_obstacle(np.array([0, 0]), np.array([0, self.scene_height]))

        # 右墙 (带出口)
        self.model.add_obstacle(np.array([self.scene_width, 0]), np.array([self.scene_width, 8]))
        self.model.add_obstacle(np.array([self.scene_width, 12]), np.array([self.scene_width, 28]))
        self.model.add_obstacle(np.array([self.scene_width, 32]), np.array([self.scene_width, self.scene_height]))

        # 闸机隔板
        barrier_y_positions = [7, 12.5, 17.5, 22.5, 27.5, 33]
        for by in barrier_y_positions:
            self.model.add_obstacle(np.array([18.5, by]), np.array([21.5, by]))

        # 柱子
        for pos in self.pillars:
            size = 0.8
            self.model.add_obstacle(pos - np.array([size, 0]), pos + np.array([size, 0]))
            self.model.add_obstacle(pos - np.array([0, size]), pos + np.array([0, size]))

        # 设施障碍
        for facility in self.facilities:
            pos = facility['position']
            w, h = facility['size']
            self.model.add_obstacle(np.array([pos[0] - w/2, pos[1] - h/2]), np.array([pos[0] + w/2, pos[1] - h/2]))
            self.model.add_obstacle(np.array([pos[0] - w/2, pos[1] + h/2]), np.array([pos[0] + w/2, pos[1] + h/2]))
            self.model.add_obstacle(np.array([pos[0] - w/2, pos[1] - h/2]), np.array([pos[0] - w/2, pos[1] + h/2]))
            self.model.add_obstacle(np.array([pos[0] + w/2, pos[1] - h/2]), np.array([pos[0] + w/2, pos[1] + h/2]))

        # 准备类型分布
        types = list(self.type_distribution.keys())
        probs = list(self.type_distribution.values())
        total = sum(probs)
        probs = [p / total for p in probs]

        # 添加行人 (使用类型分布)
        np.random.seed(None)
        for i in range(self.n_pedestrians):
            position = np.array([
                np.random.uniform(2, 14),
                np.random.uniform(12, 28)
            ])
            target = self._choose_initial_exit(position)

            # 随机选择行人类型
            ped_type = np.random.choice(types, p=probs)

            # 使用工厂方法创建带类型的行人
            ped = Pedestrian.create_with_type(
                id=i,
                position=position,
                velocity=np.zeros(2),
                target=target.copy(),
                ped_type=ped_type,
                speed_variation=True
            )
            self.model.add_pedestrian(ped)

        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.evacuated_by_type = {t: 0 for t in PedestrianType}
        self.step_count = 0
        self.current_action = 0

        # 重置引导统计
        self.total_guided_count = 0
        self.current_step_guided = 0
        self.corner_avoided_count = 0
        self.guidance_stats = {
            'by_reason': {'congestion': 0, 'corner_trap': 0},
            'by_exit': {'A': 0, 'B': 0, 'C': 0}
        }

    def _choose_initial_exit(self, position):
        """初始出口选择（无PPO时使用）"""
        y = position[1]
        if y < 15:
            weights = [0.6, 0.2, 0.2]
        elif y > 25:
            weights = [0.2, 0.6, 0.2]
        else:
            weights = [0.2, 0.2, 0.6]
        choice = np.random.choice([0, 1, 2], p=weights)
        return self.exits[choice]['position'].copy()

    def get_ppo_observation(self):
        """获取PPO模型的观测状态

        - 增强版模型 (16维): 完整观测信息
        - 旧版地铁站模型 (8维): 兼容模式
        - 更旧版模型 (6维): 兼容2出口模式
        """
        # 计算各出口附近的密度和拥堵度
        exit_densities = []
        exit_congestions = []
        exit_flow_ratios = []

        total_peds = len(self.model.pedestrians)

        for exit_info in self.exits:
            density, congestion = self._compute_exit_metrics(exit_info['position'])
            exit_densities.append(density)
            exit_congestions.append(congestion)

            # 计算走向该出口的行人比例
            flow_count = 0
            for ped in self.model.pedestrians:
                if np.linalg.norm(ped.target - exit_info['position']) < 2.0:
                    flow_count += 1
            flow_ratio = flow_count / max(total_peds, 1)
            exit_flow_ratios.append(flow_ratio)

        # 剩余人数比例
        remaining_ratio = total_peds / max(self.n_pedestrians, 1)

        # 时间比例
        time_ratio = min(self.step_count / 1000, 1.0)

        # 历史疏散速率 (简化: 使用当前疏散比例的变化)
        evac_rate = self.evacuated_count / max(self.n_pedestrians, 1)
        evacuation_rates = [evac_rate, evac_rate, evac_rate]  # 简化处理

        # 瓶颈点密度
        bottleneck_densities = self._compute_bottleneck_densities()

        # 检测模型期望的观测维度
        if self.ppo_model is not None:
            expected_dim = self.ppo_model.observation_space.shape[0]
        else:
            expected_dim = 16 if self.is_metro_model else 8

        if expected_dim >= 16:
            # 16维增强观测
            obs = np.array([
                exit_densities[0], exit_densities[1], exit_densities[2],
                exit_congestions[0], exit_congestions[1], exit_congestions[2],
                exit_flow_ratios[0], exit_flow_ratios[1], exit_flow_ratios[2],
                evacuation_rates[0], evacuation_rates[1], evacuation_rates[2],
                bottleneck_densities[0], bottleneck_densities[1],
                remaining_ratio, time_ratio
            ], dtype=np.float32)
        elif expected_dim == 8 or self.is_metro_model:
            # 8维观测: 3出口密度 + 3出口拥堵度 + 剩余比例 + 时间比例
            obs = np.array([
                exit_densities[0], exit_densities[1], exit_densities[2],
                exit_congestions[0], exit_congestions[1], exit_congestions[2],
                remaining_ratio, time_ratio
            ], dtype=np.float32)
        else:
            # 6维观测 (兼容旧版2出口模型)
            obs = np.array([
                exit_densities[0], exit_densities[1],
                exit_congestions[0], exit_congestions[1],
                remaining_ratio, time_ratio
            ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _compute_bottleneck_densities(self):
        """计算瓶颈点密度"""
        gate_zone = {'x_min': 18, 'x_max': 22, 'y_min': 7, 'y_max': 33}
        gate_count = 0
        pillar_count = 0

        for ped in self.model.pedestrians:
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

        # 归一化
        gate_density = min(gate_count / 20.0, 1.0)
        pillar_density = min(pillar_count / 20.0, 1.0)

        return [gate_density, pillar_density]

    def _compute_exit_metrics(self, exit_pos):
        """计算出口附近的密度和拥堵度"""
        radius = 8.0
        nearby_peds = []

        for ped in self.model.pedestrians:
            dist = np.linalg.norm(ped.position - exit_pos)
            if dist < radius:
                nearby_peds.append(ped)

        density = min(len(nearby_peds) / 25.0, 1.0)

        if len(nearby_peds) > 0:
            avg_speed = np.mean([ped.speed for ped in nearby_peds])
            congestion = max(0, 1 - avg_speed / 1.2)
        else:
            congestion = 0.0

        return density, congestion

    def apply_ppo_guidance(self):
        """应用PPO引导策略（增强版：考虑拥堵情况）

        - 地铁站模型: action 0=A, 1=B, 2=C (直接映射)
        - 旧版模型: action 0=A, 1=B (需要额外逻辑处理出口C)
        - 增强: 如果推荐出口太拥堵，自动选择替代出口
        """
        if self.ppo_model is None:
            return

        # 获取观测
        obs = self.get_ppo_observation()

        # PPO决策
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        self.current_action = int(action)

        if self.is_metro_model:
            # 地铁站模型: 直接使用3出口映射 (0=A, 1=B, 2=C)
            recommended_exit = min(self.current_action, 2)
        else:
            # 旧版模型: 只有2个动作 (0=A, 1=B)
            recommended_exit = self.current_action

            # 检查推荐出口的拥堵情况，如果拥堵则考虑出口C
            _, congestion = self._compute_exit_metrics(self.exits[recommended_exit]['position'])
            if congestion > 0.6:
                recommended_exit = 2  # 拥堵时引导去出口C

        # 增强：检查推荐出口的拥堵情况，如果太拥堵则选择最不拥堵的出口
        _, rec_congestion = self._compute_exit_metrics(self.exits[recommended_exit]['position'])
        if rec_congestion > 0.5:
            # 找到最不拥堵的出口
            best_exit = recommended_exit
            min_congestion = rec_congestion
            for i, exit_info in enumerate(self.exits):
                _, cong = self._compute_exit_metrics(exit_info['position'])
                if cong < min_congestion:
                    min_congestion = cong
                    best_exit = i
            recommended_exit = best_exit
            self.current_action = best_exit  # 更新显示

        # 引导部分行人改变目标
        target_pos = self.exits[recommended_exit]['position']

        for ped in self.model.pedestrians:
            # 只有在大厅区域的行人才响应引导（已过闸机）
            if ped.position[0] > 22:
                # 根据距离和随机因素决定是否听从引导
                dist_to_current = np.linalg.norm(ped.position - ped.target)
                dist_to_recommended = np.linalg.norm(ped.position - target_pos)

                # 如果推荐出口更近或差距不大，更有可能响应
                if dist_to_recommended < dist_to_current * 1.3:
                    prob = 0.15  # 降低基础响应概率，避免所有人去同一出口
                else:
                    prob = 0.05

                # 如果推荐出口不拥堵，提高响应概率
                _, congestion = self._compute_exit_metrics(target_pos)
                if congestion < 0.3:
                    prob *= 2.0
                elif congestion > 0.6:
                    prob *= 0.3  # 拥堵时大幅降低响应概率

                if np.random.random() < prob:
                    ped.target = target_pos.copy()

    def draw_scene(self, screen, font, font_small):
        """绘制场景"""
        screen.fill(BG_COLOR)

        # 绘制地面区域
        platform_rect = pygame.Rect(
            self.world_to_screen(np.array([0, 30]))[0],
            self.world_to_screen(np.array([0, 30]))[1],
            int(15 * self.scale), int(20 * self.scale)
        )
        pygame.draw.rect(screen, (200, 180, 160), platform_rect)

        hall_rect = pygame.Rect(
            self.world_to_screen(np.array([25, 40]))[0],
            self.world_to_screen(np.array([25, 40]))[1],
            int(35 * self.scale), int(40 * self.scale)
        )
        pygame.draw.rect(screen, FLOOR_COLOR, hall_rect)

        # 绘制PPO推荐出口高亮
        if self.ppo_model is not None and self.use_ppo:
            recommended = self.exits[min(self.current_action, 2)]
            pos = recommended['position']
            width = recommended['width']

            if pos[0] >= self.scene_width - 0.1:
                highlight_pos = self.world_to_screen(np.array([self.scene_width - 2, pos[1]]))
                pygame.draw.circle(screen, PURPLE, highlight_pos, int(3 * self.scale), 3)
            else:
                highlight_pos = self.world_to_screen(np.array([pos[0], self.scene_height - 2]))
                pygame.draw.circle(screen, PURPLE, highlight_pos, int(3 * self.scale), 3)

        # 绘制出口区域
        for exit_info in self.exits:
            pos = exit_info['position']
            width = exit_info['width']

            if pos[0] >= self.scene_width - 0.1:
                top_left = self.world_to_screen(np.array([self.scene_width - 1, pos[1] + width/2]))
                rect_w = int(2 * self.scale)
                rect_h = int(width * self.scale)
            else:
                top_left = self.world_to_screen(np.array([pos[0] - width/2, self.scene_height]))
                rect_w = int(width * self.scale)
                rect_h = int(2 * self.scale)

            pygame.draw.rect(screen, LIGHT_GREEN, (top_left[0], top_left[1], rect_w, rect_h))
            label = font.render(exit_info['label'], True, GREEN)
            screen.blit(label, (top_left[0] + 5, top_left[1] + 5))

        # 绘制闸机
        for gate in self.gates:
            pos = gate['position']
            screen_pos = self.world_to_screen(pos)
            gate_w = int(gate['length'] * self.scale)
            gate_h = int(1.5 * self.scale)
            rect = pygame.Rect(screen_pos[0] - gate_w//2, screen_pos[1] - gate_h//2, gate_w, gate_h)
            pygame.draw.rect(screen, GATE_COLOR, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)
            pygame.draw.polygon(screen, YELLOW, [
                (rect.centerx - 5, rect.centery - 3),
                (rect.centerx + 5, rect.centery),
                (rect.centerx - 5, rect.centery + 3),
            ])

        # 绘制柱子
        for pos in self.pillars:
            screen_pos = self.world_to_screen(pos)
            size = int(1.2 * self.scale)
            rect = pygame.Rect(screen_pos[0] - size//2, screen_pos[1] - size//2, size, size)
            pygame.draw.rect(screen, GRAY, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)

        # 绘制设施
        for facility in self.facilities:
            pos = facility['position']
            w, h = facility['size']
            screen_pos = self.world_to_screen(pos)
            rect_w, rect_h = int(w * self.scale), int(h * self.scale)
            rect = pygame.Rect(screen_pos[0] - rect_w//2, screen_pos[1] - rect_h//2, rect_w, rect_h)
            color = (100, 150, 200) if facility['type'] == 'info' else (180, 180, 180)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)
            label = font_small.render(facility['label'], True, BLACK)
            screen.blit(label, (rect.centerx - label.get_width()//2, rect.centery - label.get_height()//2))

        # 绘制墙壁
        walls = [
            (np.array([0, self.scene_height]), np.array([35, self.scene_height])),
            (np.array([45, self.scene_height]), np.array([self.scene_width, self.scene_height])),
            (np.array([0, 0]), np.array([self.scene_width, 0])),
            (np.array([0, 0]), np.array([0, self.scene_height])),
            (np.array([self.scene_width, 0]), np.array([self.scene_width, 8])),
            (np.array([self.scene_width, 12]), np.array([self.scene_width, 28])),
            (np.array([self.scene_width, 32]), np.array([self.scene_width, self.scene_height])),
        ]
        for start, end in walls:
            pygame.draw.line(screen, BLACK, self.world_to_screen(start), self.world_to_screen(end), 4)

        # 绘制角落陷阱区域（警告区域）
        for corner in self.corner_traps:
            corner_screen = self.world_to_screen(corner)
            trap_radius = int(3.0 * self.scale)
            # 半透明红色圆圈
            s = pygame.Surface((trap_radius * 2, trap_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 100, 100, 50), (trap_radius, trap_radius), trap_radius)
            screen.blit(s, (corner_screen[0] - trap_radius, corner_screen[1] - trap_radius))

        # 绘制预测轨迹（如果启用）
        if self.show_predictions and self.predicted_trajectories:
            for ped_id, pred_traj in self.predicted_trajectories.items():
                if len(pred_traj) < 2:
                    continue

                # 绘制预测轨迹线
                points = []
                for pos in pred_traj[::2]:  # 每隔一个点绘制，减少视觉混乱
                    screen_pos = self.world_to_screen(pos)
                    points.append(screen_pos)

                if len(points) >= 2:
                    # 使用半透明的线
                    pygame.draw.lines(screen, PREDICTION_COLOR, False, points, 1)

                    # 在轨迹终点绘制小圆点
                    end_pos = self.world_to_screen(pred_traj[-1])
                    pygame.draw.circle(screen, PREDICTION_COLOR, end_pos, 3)

        # 绘制行人
        for ped in self.model.pedestrians:
            screen_pos = self.world_to_screen(ped.position)
            radius = int(0.35 * self.scale)

            if self.show_type_colors:
                # 按行人类型着色
                base_color = PEDESTRIAN_TYPE_COLORS.get(ped.ped_type, BLUE)

                # 如果行人在等待或恐慌，修改颜色
                if ped.is_waiting:
                    # 等待时显示更暗的颜色
                    color = tuple(max(0, c - 50) for c in base_color)
                elif ped.panic_factor > 0.2:
                    # 恐慌时添加红色调
                    r = min(255, base_color[0] + int(100 * ped.panic_factor))
                    color = (r, base_color[1], base_color[2])
                else:
                    color = base_color
            else:
                # 按速度着色 (原始逻辑)
                speed = ped.speed
                if speed < 0.3:
                    color = RED
                elif speed < 0.8:
                    color = ORANGE
                else:
                    color = BLUE

            pygame.draw.circle(screen, color, screen_pos, radius)

            # 边框颜色 (等待时用白色突出显示)
            border_color = WHITE if ped.is_waiting else DARK_BLUE
            pygame.draw.circle(screen, border_color, screen_pos, radius, 1)

        # Area labels
        platform_label = font.render("Platform", True, (100, 80, 60))
        screen.blit(platform_label, self.world_to_screen(np.array([3, 22])))
        gate_label = font.render("Gate", True, BLACK)
        screen.blit(gate_label, self.world_to_screen(np.array([18, 36])))
        hall_label = font.render("Exit Hall", True, (80, 80, 80))
        screen.blit(hall_label, self.world_to_screen(np.array([38, 22])))

        # 信息面板
        self.draw_info_panel(screen, font, font_small)

    def draw_info_panel(self, screen, font, font_small):
        """绘制信息面板（使用英文显示避免编码问题）"""
        panel_x = int(self.scene_width * self.scale) + 70
        panel_y = 20

        time_elapsed = self.step_count * self.dt * 3
        remaining = len(self.model.pedestrians)

        # Title
        title = font.render("Metro Station Exit", True, BLACK)
        screen.blit(title, (panel_x, panel_y))

        # PPO status
        if self.ppo_model is not None and self.use_ppo:
            model_type = "3-exit" if self.is_metro_model else "2-exit"
            ppo_status = font_small.render(f"PPO Guidance: ON ({model_type})", True, PURPLE)
            recommended = self.exits[min(self.current_action, 2)]['name']
            ppo_action = font_small.render(f"Recommend: Exit {recommended}", True, PURPLE)
        else:
            ppo_status = font_small.render("PPO Guidance: OFF", True, GRAY)
            ppo_action = font_small.render("(Press P to enable)", True, GRAY)

        screen.blit(ppo_status, (panel_x, panel_y + 25))
        screen.blit(ppo_action, (panel_x, panel_y + 45))

        # GBM status
        if self.gbm_loaded:
            gbm_status = font_small.render("GBM Predictor: ON", True, GREEN)
        else:
            gbm_status = font_small.render("GBM Predictor: OFF", True, GRAY)
        screen.blit(gbm_status, (panel_x, panel_y + 65))

        # Trajectory predictor status
        if self.trajectory_loaded:
            if self.trajectory_predictor and self.trajectory_predictor.use_neural_network:
                actual_type = self.trajectory_predictor.actual_model_type
                if actual_type == 'trajectron':
                    traj_status = font_small.render("Trajectron++: ON", True, CYAN)
                else:
                    traj_status = font_small.render("Social-LSTM: ON", True, CYAN)
            else:
                traj_status = font_small.render("Trajectory: Linear", True, YELLOW)
        else:
            traj_status = font_small.render("Trajectory: OFF", True, GRAY)
        screen.blit(traj_status, (panel_x, panel_y + 85))

        # Guidance system status
        guidance_status = font_small.render(f"Guided Total: {self.total_guided_count}", True, PURPLE)
        screen.blit(guidance_status, (panel_x, panel_y + 105))

        # Corner avoidance status
        if self.corner_avoided_count > 0:
            corner_status = font_small.render(f"Corner Avoided: {self.corner_avoided_count}", True, PINK)
        else:
            corner_status = font_small.render("Corner Avoided: 0", True, GRAY)
        screen.blit(corner_status, (panel_x, panel_y + 125))

        # Enhanced behavior status
        if self.enable_enhanced_behaviors:
            behavior_status = font_small.render("Enhanced Behavior: ON", True, GREEN)
        else:
            behavior_status = font_small.render("Enhanced Behavior: OFF", True, GRAY)
        screen.blit(behavior_status, (panel_x, panel_y + 145))

        lines = [
            "",
            f"Time: {time_elapsed:.1f}s",
            f"Evacuated: {self.evacuated_count}",
            f"Remaining: {remaining}",
            f"Total: {self.n_pedestrians}",
            "",
            "By Exit:",
            f"  Exit A: {self.evacuated_by_exit['A']}",
            f"  Exit B: {self.evacuated_by_exit['B']}",
            f"  Exit C: {self.evacuated_by_exit['C']}",
            "",
            "By Type:",
            f"  Normal: {self.evacuated_by_type.get(PedestrianType.NORMAL, 0)}",
            f"  Elderly: {self.evacuated_by_type.get(PedestrianType.ELDERLY, 0)}",
            f"  Child: {self.evacuated_by_type.get(PedestrianType.CHILD, 0)}",
            f"  Impatient: {self.evacuated_by_type.get(PedestrianType.IMPATIENT, 0)}",
            "",
            "== Controls ==",
            "SPACE: Pause/Resume",
            "P: Toggle PPO",
            "T: Toggle color mode",
            "V: Toggle predictions",
            "R: Restart",
            "ESC: Exit",
        ]

        for i, line in enumerate(lines):
            text = font_small.render(line, True, BLACK)
            screen.blit(text, (panel_x, panel_y + 165 + i * 16))

        # Legend
        legend_y = panel_y + 165 + len(lines) * 16 + 10

        legend_title = font_small.render("== Legend ==", True, BLACK)
        screen.blit(legend_title, (panel_x, legend_y))
        legend_y += 20

        if self.show_type_colors:
            # Type color legend
            legends = [
                (PEDESTRIAN_TYPE_COLORS[PedestrianType.NORMAL], "Normal Adult"),
                (PEDESTRIAN_TYPE_COLORS[PedestrianType.ELDERLY], "Elderly"),
                (PEDESTRIAN_TYPE_COLORS[PedestrianType.CHILD], "Child"),
                (PEDESTRIAN_TYPE_COLORS[PedestrianType.IMPATIENT], "Impatient"),
                (PURPLE, "PPO Recommended"),
                (PREDICTION_COLOR, "Pred. Trajectory"),
            ]
        else:
            # Speed color legend
            legends = [
                (BLUE, "Normal Speed"),
                (ORANGE, "Slow"),
                (RED, "Congested"),
                (PURPLE, "PPO Recommended"),
                (PREDICTION_COLOR, "Pred. Trajectory"),
            ]

        for i, (color, label) in enumerate(legends):
            pygame.draw.circle(screen, color, (panel_x + 10, legend_y + i * 18 + 6), 5)
            text = font_small.render(label, True, BLACK)
            screen.blit(text, (panel_x + 25, legend_y + i * 18))

    def predict_and_rebalance(self):
        """预测性疏通系统：
        1. 使用神经网络预测轨迹
        2. 检测角落陷阱并重定向
        3. 将过载出口的行人重分配到其他出口
        """
        if len(self.model.pedestrians) < 5:
            return 0

        redirect_count = 0
        self.corner_avoided_count = 0

        # 1. 更新轨迹预测
        if self.trajectory_predictor is not None:
            # 更新历史
            for ped in self.model.pedestrians:
                self.trajectory_predictor.update_history(ped.id, ped.position)

            # 批量预测
            self.predicted_trajectories = self.trajectory_predictor.predict_all_trajectories(
                self.model.pedestrians,
                scene_bounds=(0, 0, self.scene_width, self.scene_height)
            )

            # 2. 角落陷阱检测和避免
            for ped in self.model.pedestrians:
                if ped.id not in self.predicted_trajectories:
                    continue

                pred_traj = self.predicted_trajectories[ped.id]
                is_trapped, trap_corner = self.trajectory_predictor.detect_corner_trap(
                    pred_traj, self.corner_traps, trap_radius=3.0
                )

                if is_trapped:
                    # 重定向到远离陷阱的出口
                    best_exit = None
                    best_score = float('-inf')

                    for exit_info in self.exits:
                        exit_pos = exit_info['position']
                        dist_to_trap = np.linalg.norm(exit_pos - trap_corner)
                        dist_to_ped = np.linalg.norm(exit_pos - ped.position)

                        # 分数：远离陷阱 + 不太远
                        score = dist_to_trap - dist_to_ped * 0.3

                        if score > best_score:
                            best_score = score
                            best_exit = exit_info

                    if best_exit is not None:
                        ped.target = best_exit['position'].copy()
                        self.corner_avoided_count += 1
                        redirect_count += 1

        # 3. 出口负载均衡
        threshold = 8  # 单出口人数阈值

        # 统计每个出口附近的行人数量
        exit_counts = {}
        exit_peds = {}
        for exit_info in self.exits:
            exit_name = exit_info['name']
            exit_pos = exit_info['position']
            exit_counts[exit_name] = 0
            exit_peds[exit_name] = []

            for ped in self.model.pedestrians:
                dist_to_exit = np.linalg.norm(ped.target - exit_pos)
                if dist_to_exit < 1.0:
                    exit_counts[exit_name] += 1
                    current_dist = np.linalg.norm(ped.position - exit_pos)
                    if current_dist > 6.0:
                        exit_peds[exit_name].append(ped)

        # 重分配过载出口的行人
        for exit_info in self.exits:
            exit_name = exit_info['name']
            if exit_counts[exit_name] <= threshold:
                continue

            excess = exit_counts[exit_name] - threshold
            candidates = exit_peds[exit_name]

            if not candidates:
                continue

            alternatives = [e for e in self.exits
                          if exit_counts[e['name']] < threshold]

            if not alternatives:
                continue

            redirected = 0
            for ped in candidates:
                if redirected >= excess:
                    break

                best_alt = None
                best_dist = float('inf')
                for alt in alternatives:
                    dist = np.linalg.norm(ped.position - alt['position'])
                    if dist < best_dist:
                        best_dist = dist
                        best_alt = alt

                if best_alt and best_dist < 50:
                    ped.target = best_alt['position'].copy()
                    redirected += 1
                    redirect_count += 1

        return redirect_count

    # ========== 分层预测式引导系统 ==========

    def predictive_guidance_system(self) -> int:
        """分层预测式引导系统

        第1层：PPO全局决策 - 获取推荐出口
        第2层：Social-LSTM预测筛选 - 识别将遇到问题的行人
        第3层：个体决策 - 检查引导条件后引导

        Returns:
            本次引导的行人数量
        """
        if self.trajectory_predictor is None:
            return 0

        guided_count = 0
        current_time = self.step_count * self.dt

        # 第1层：PPO推荐的出口
        recommended_exit = self.exits[min(self.current_action, 2)]

        # 第2层：预测所有人轨迹，识别问题行人
        problem_pedestrians = self._identify_problem_pedestrians()

        # 第3层：对问题行人进行个体引导决策
        for ped in problem_pedestrians:
            if self._can_be_guided(ped, current_time):
                best_exit = self._find_best_alternative_exit(ped, recommended_exit)
                if best_exit is not None:
                    self._apply_guidance(ped, best_exit, current_time)
                    guided_count += 1

        self.current_step_guided = guided_count
        self.total_guided_count += guided_count
        return guided_count

    def _identify_problem_pedestrians(self):
        """识别将遇到问题的行人（主动预防式）

        问题类型（优先级从高到低）：
        1. 走向角落陷阱
        2. 走向已拥堵的出口
        3. 走向过载出口（负载不均衡）

        关键：只从过载出口分流行人
        """
        problem_peds = []
        cfg = GUIDANCE_CONFIG

        # 统计每个出口的目标人数
        exit_target_count = {exit_info['id']: 0 for exit_info in self.exits}

        for ped in self.model.pedestrians:
            for exit_info in self.exits:
                if np.linalg.norm(ped.target - exit_info['position']) < 2.0:
                    exit_target_count[exit_info['id']] += 1
                    break

        total_peds = len(self.model.pedestrians)
        if total_peds < cfg['min_peds_for_rebalance']:
            return problem_peds

        # 找出过载出口
        avg_per_exit = total_peds / len(self.exits)
        overloaded_exit_ids = set()

        for exit_info in self.exits:
            count = exit_target_count[exit_info['id']]
            ratio = count / total_peds if total_peds > 0 else 0
            if ratio > cfg['exit_imbalance_threshold'] and count > avg_per_exit * 1.2:
                overloaded_exit_ids.add(exit_info['id'])

        # 识别问题行人
        for ped in self.model.pedestrians:
            if ped.id not in self.predicted_trajectories:
                continue

            pred_traj = self.predicted_trajectories[ped.id]

            # 检查1：走向角落陷阱
            is_trapped, _ = self.trajectory_predictor.detect_corner_trap(
                pred_traj, self.corner_traps, trap_radius=cfg['corner_trap_radius']
            )
            if is_trapped:
                problem_peds.append(ped)
                continue

            # 检查2：走向已拥堵的出口
            if self._will_reach_congested_exit(ped):
                problem_peds.append(ped)
                continue

            # 检查3：走向过载出口
            for exit_info in self.exits:
                if exit_info['id'] in overloaded_exit_ids:
                    if np.linalg.norm(ped.target - exit_info['position']) < 2.0:
                        dist_to_exit = np.linalg.norm(ped.position - exit_info['position'])
                        rebalance_dist = cfg.get('rebalance_distance_threshold', 10.0)
                        if dist_to_exit > rebalance_dist:
                            problem_peds.append(ped)
                        break

        return problem_peds

    def _will_reach_congested_exit(self, ped) -> bool:
        """检查行人是否正在走向拥堵出口"""
        cfg = GUIDANCE_CONFIG

        # 找到行人当前目标对应的出口
        target_exit = None
        min_dist = float('inf')
        for exit_info in self.exits:
            dist = np.linalg.norm(ped.target - exit_info['position'])
            if dist < min_dist:
                min_dist = dist
                target_exit = exit_info

        if target_exit is None:
            return False

        # 计算该出口的拥堵度
        _, congestion = self._compute_exit_metrics(target_exit['position'])

        return congestion > cfg['congestion_threshold']

    def _can_be_guided(self, ped, current_time: float) -> bool:
        """检查行人是否可以被引导"""
        cfg = GUIDANCE_CONFIG

        # 条件1：在引导区域内（已过闸机）
        if ped.position[0] <= cfg['guidance_zone_x']:
            return False

        # 条件2：引导次数未超限
        if ped.guidance_count >= cfg['max_guidance_count']:
            return False

        # 条件3：冷却时间已过
        if current_time - ped.last_guidance_time < cfg['cooldown_time']:
            return False

        # 条件4：距离目标足够远
        dist_to_target = np.linalg.norm(ped.position - ped.target)
        if dist_to_target < cfg['min_distance_to_target']:
            return False

        return True

    def _find_best_alternative_exit(self, ped, recommended_exit):
        """为问题行人找到最佳替代出口

        核心原则：
        1. 不引导到过载出口（目标人数过多的出口）
        2. 优先引导到人少且近的出口
        3. 只有明显更优时才改道
        """
        cfg = GUIDANCE_CONFIG

        # 当前目标出口及距离
        current_target_exit = None
        current_dist = float('inf')
        for exit_info in self.exits:
            if np.linalg.norm(ped.target - exit_info['position']) < 1.0:
                current_target_exit = exit_info
                current_dist = np.linalg.norm(ped.position - exit_info['position'])
                break

        if current_target_exit is None:
            return None

        # 统计各出口目标人数，确定过载出口
        exit_target_count = {exit_info['id']: 0 for exit_info in self.exits}
        for p in self.model.pedestrians:
            for exit_info in self.exits:
                if np.linalg.norm(p.target - exit_info['position']) < 2.0:
                    exit_target_count[exit_info['id']] += 1
                    break

        # 计算当前出口的拥堵度
        _, current_congestion = self._compute_exit_metrics(current_target_exit['position'])
        current_count = exit_target_count[current_target_exit['id']]

        # 寻找最佳替代出口
        best_exit = None
        best_score = float('-inf')

        for exit_info in self.exits:
            if exit_info['id'] == current_target_exit['id']:
                continue

            # 关键检查：不引导到人更多的出口
            target_count = exit_target_count[exit_info['id']]
            if target_count >= current_count:
                continue  # 新出口人不比当前少，不考虑

            _, congestion = self._compute_exit_metrics(exit_info['position'])
            dist_to_exit = np.linalg.norm(ped.position - exit_info['position'])

            # 评分：人数差距 + 距离因素
            # 人数差距更重要（权重更高）
            count_benefit = (current_count - target_count) * 5  # 每少1人 = +5分
            distance_cost = (dist_to_exit - current_dist) * 0.5  # 每远1米 = -0.5分
            congestion_benefit = (current_congestion - congestion) * 10

            score = count_benefit + congestion_benefit - distance_cost

            # 只有分数为正（确实更好）才考虑
            if score > 0 and score > best_score:
                best_score = score
                best_exit = exit_info

        return best_exit

    def _apply_guidance(self, ped, new_exit, current_time: float) -> None:
        """应用引导并更新行人状态"""
        # 记录原始目标
        if ped.original_target is None:
            ped.original_target = ped.target.copy()

        # 更新目标
        ped.target = new_exit['position'].copy()

        # 更新引导状态
        ped.guidance_count += 1
        ped.last_guidance_time = current_time

        # 统计
        self.guidance_stats['by_exit'][new_exit['name']] += 1

    def _avoid_corner_traps(self) -> int:
        """角落陷阱避免（作为分层引导的补充）"""
        if self.trajectory_predictor is None:
            return 0

        avoided_count = 0
        current_time = self.step_count * self.dt
        cfg = GUIDANCE_CONFIG

        for ped in self.model.pedestrians:
            if ped.id not in self.predicted_trajectories:
                continue

            # 检查是否可以被引导（避免重复引导）
            if not self._can_be_guided(ped, current_time):
                continue

            pred_traj = self.predicted_trajectories[ped.id]
            is_trapped, trap_corner = self.trajectory_predictor.detect_corner_trap(
                pred_traj, self.corner_traps, trap_radius=cfg['corner_trap_radius']
            )

            if is_trapped:
                # 找到远离陷阱的出口
                best_exit = None
                best_score = float('-inf')

                for exit_info in self.exits:
                    exit_pos = exit_info['position']
                    dist_to_trap = np.linalg.norm(exit_pos - trap_corner)
                    dist_to_ped = np.linalg.norm(exit_pos - ped.position)
                    score = dist_to_trap - dist_to_ped * 0.3

                    if score > best_score:
                        best_score = score
                        best_exit = exit_info

                if best_exit is not None:
                    self._apply_guidance(ped, best_exit, current_time)
                    self.guidance_stats['by_reason']['corner_trap'] += 1
                    avoided_count += 1

        self.corner_avoided_count = avoided_count
        return avoided_count

    def update(self):
        """更新模拟"""
        if self.paused:
            return

        # ========== 分层预测式引导系统 ==========
        # 替代原有的随机概率引导
        self.current_step_guided = 0

        if self.step_count % 5 == 0 and len(self.model.pedestrians) > 5:
            # 更新PPO决策（获取推荐出口）
            if self.ppo_model is not None and self.use_ppo:
                obs = self.get_ppo_observation()
                action, _ = self.ppo_model.predict(obs, deterministic=True)
                self.current_action = int(action)

            # 更新轨迹预测
            if self.trajectory_predictor is not None:
                for ped in self.model.pedestrians:
                    self.trajectory_predictor.update_history(ped.id, ped.position)
                self.predicted_trajectories = self.trajectory_predictor.predict_all_trajectories(
                    self.model.pedestrians,
                    scene_bounds=(0, 0, self.scene_width, self.scene_height)
                )

                # 使用分层预测式引导系统
                self.predictive_guidance_system()

                # 角落陷阱避免（补充）
                self._avoid_corner_traps()
            else:
                # 回退到旧版负载均衡
                self.predict_and_rebalance()

        # 物理模拟
        for _ in range(3):
            self.model.step(self.dt)

        # 检查疏散
        to_remove = []
        for ped in self.model.pedestrians:
            for exit_info in self.exits:
                dist = np.linalg.norm(ped.position - exit_info['position'])
                if dist < exit_info['width']:
                    to_remove.append((ped, exit_info['name']))
                    break

        for ped, exit_name in to_remove:
            self.model.pedestrians.remove(ped)
            self.evacuated_count += 1
            self.evacuated_by_exit[exit_name] += 1
            # 按类型统计
            self.evacuated_by_type[ped.ped_type] += 1
            # 清理轨迹预测历史
            if self.trajectory_predictor is not None:
                self.trajectory_predictor.remove_pedestrian(ped.id)
            # 清理预测缓存
            if ped.id in self.predicted_trajectories:
                del self.predicted_trajectories[ped.id]

        self.step_count += 1

    def run(self):
        """运行可视化"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Metro Evacuation - SFM + PPO + GBM")
        clock = pygame.time.Clock()

        # 使用Arial字体避免中文编码问题
        font = pygame.font.SysFont('Arial', 18)
        font_small = pygame.font.SysFont('Arial', 14)

        # Load PPO model, GBM model, and Trajectory predictor
        self.load_ppo_model()
        self.load_gbm_predictor()
        self.load_trajectory_predictor()
        self.setup_model()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.setup_model()
                    elif event.key == pygame.K_p:
                        # Toggle PPO
                        if self.ppo_model is not None:
                            self.use_ppo = not self.use_ppo
                            print(f"PPO Guidance: {'ON' if self.use_ppo else 'OFF'}")
                    elif event.key == pygame.K_t:
                        # Toggle color mode (by type/by speed)
                        self.show_type_colors = not self.show_type_colors
                        mode = "by Type" if self.show_type_colors else "by Speed"
                        print(f"Color Mode: {mode}")
                    elif event.key == pygame.K_v:
                        # Toggle prediction visualization
                        self.show_predictions = not self.show_predictions
                        print(f"Show Predictions: {'ON' if self.show_predictions else 'OFF'}")

            self.update()
            self.draw_scene(screen, font, font_small)
            pygame.display.flip()
            clock.tick(60)

            if len(self.model.pedestrians) == 0 and not self.paused:
                self.paused = True
                time_elapsed = self.step_count * self.dt * 3
                print(f"\nEvacuation Complete!")
                print(f"Total Time: {time_elapsed:.1f}s")
                print(f"By Exit: A={self.evacuated_by_exit['A']}, "
                      f"B={self.evacuated_by_exit['B']}, C={self.evacuated_by_exit['C']}")
                print(f"By Type: Normal={self.evacuated_by_type.get(PedestrianType.NORMAL, 0)}, "
                      f"Elderly={self.evacuated_by_type.get(PedestrianType.ELDERLY, 0)}, "
                      f"Child={self.evacuated_by_type.get(PedestrianType.CHILD, 0)}, "
                      f"Impatient={self.evacuated_by_type.get(PedestrianType.IMPATIENT, 0)}")
                print(f"PPO Guidance: {'ON' if self.use_ppo else 'OFF'}")
                print(f"Enhanced Behaviors: {'ON' if self.enable_enhanced_behaviors else 'OFF'}")
                print(f"== Hierarchical Guidance Stats ==")
                print(f"Total Guided: {self.total_guided_count}")
                print(f"Guided to Exit: A={self.guidance_stats['by_exit']['A']}, "
                      f"B={self.guidance_stats['by_exit']['B']}, C={self.guidance_stats['by_exit']['C']}")

        pygame.quit()


def main():
    print("=" * 60)
    print("Metro Station Evacuation Simulation - Hierarchical Guidance")
    print("SFM + PPO + GBM + Trajectron++/Social-LSTM + Predictive Guidance")
    print("=" * 60)
    print("\nModels:")
    print("  - Social Force Model (SFM): Pedestrian dynamics (Helbing 1995)")
    print("  - PPO Reinforcement Learning: Global exit recommendation")
    print("  - GBM Behavior Predictor: Based on ETH/UCY dataset")
    print("  - Trajectron++: Multi-modal GNN trajectory prediction (Salzmann 2020)")
    print("  - Social-LSTM: Single-modal trajectory prediction (Alahi 2016)")
    print("  - Pedestrian Types: Normal(70%), Elderly(15%), Child(10%), Impatient(5%)")
    print("  - Enhanced Behaviors: Waiting, Hesitation, Panic")
    print("\n== Hierarchical Predictive Guidance System ==")
    print("  - Layer 1: PPO Global Decision - Recommends optimal exit")
    print("  - Layer 2: Social-LSTM Filtering - Identifies problem pedestrians")
    print("  - Layer 3: Individual Decision - Checks guidance conditions")
    print(f"  - Max guidance per person: {GUIDANCE_CONFIG['max_guidance_count']}")
    print(f"  - Cooldown time: {GUIDANCE_CONFIG['cooldown_time']}s")
    print("\nLiterature:")
    print("  - Helbing 1995: Desired speed 1.34 m/s")
    print("  - Weidmann 1993: Elderly speed 0.9 m/s")
    print("  - Fruin 1971: Child speed 0.7 m/s")
    print("  - Alahi 2016: Social LSTM for trajectory prediction")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  P     - Toggle PPO guidance")
    print("  T     - Toggle color mode (by type/by speed)")
    print("  V     - Toggle trajectory prediction visualization")
    print("  R     - Restart")
    print("  ESC   - Exit")
    print()

    visualizer = MetroStationWithPPO(
        n_pedestrians=80,
        scale=12.0,
        dt=0.05,
        use_ppo=True,
        enable_enhanced_behaviors=True,
        show_type_colors=True,
        show_predictions=True,
        enable_neural_prediction=True,
        type_distribution={
            PedestrianType.NORMAL: 0.70,
            PedestrianType.ELDERLY: 0.15,
            PedestrianType.CHILD: 0.10,
            PedestrianType.IMPATIENT: 0.05,
        }
    )
    visualizer.run()


if __name__ == "__main__":
    main()
