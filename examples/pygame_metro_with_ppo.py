"""
成都东客站地铁出站口疏散仿真 - 集成PPO智能引导
社会力模型 + PPO强化学习动态优化出口选择

增强版本:
- 支持多种行人类型可视化 (不同颜色)
- 显示GBM行为预测器状态
- 显示行人行为状态 (等待、恐慌)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pygame
from stable_baselines3 import PPO
from sfm.social_force import (
    SocialForceModel,
    Pedestrian,
    PedestrianType,
    PEDESTRIAN_TYPE_PARAMS
)


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
BG_COLOR = (240, 240, 235)
FLOOR_COLOR = (220, 220, 210)
GATE_COLOR = (100, 100, 100)

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
        type_distribution: dict = None
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
        """
        self.n_pedestrians = n_pedestrians
        self.scale = scale
        self.dt = dt
        self.use_ppo = use_ppo
        self.enable_enhanced_behaviors = enable_enhanced_behaviors
        self.show_type_colors = show_type_colors

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
                print(f"PPO模型已加载: {model_path}")
                if self.is_metro_model:
                    print("  (使用地铁站专用3出口模型，观测维度=8)")
                else:
                    print("  (使用旧版2出口模型，观测维度=6)")
                    print("  提示: 运行 python examples/train_ppo_metro.py 训练地铁站专用模型")
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
        """加载GBM行为预测模型"""
        gbm_model_path = project_root / "outputs" / "models" / "gbm_behavior.joblib"

        if gbm_model_path.exists():
            try:
                from ml.gbm_predictor import GBMPredictor
                self.gbm_predictor = GBMPredictor()
                self.gbm_predictor.load(str(gbm_model_path))
                self.gbm_loaded = True
                print(f"GBM行为预测模型已加载: {gbm_model_path}")
                return True
            except Exception as e:
                print(f"加载GBM模型失败: {e}")
                self.gbm_predictor = None
                self.gbm_loaded = False
                return False
        else:
            print("未找到GBM行为预测模型")
            print("  提示: 运行 python examples/train_gbm_behavior.py 训练模型")
            self.gbm_loaded = False
            return False

    def world_to_screen(self, pos):
        """世界坐标转屏幕坐标"""
        x = int(pos[0] * self.scale) + 50
        y = int((self.scene_height - pos[1]) * self.scale) + 30
        return (x, y)

    def setup_model(self):
        """初始化社会力模型

        增强版本:
        - 支持等待、犹豫、恐慌等行为
        - 支持多种行人类型
        """
        # Create enhanced social force model
        # Increased wall_A to prevent pedestrians getting stuck
        self.model = SocialForceModel(
            tau=0.5,
            A=2000.0,
            B=0.08,
            wall_A=5000.0,    # Increased from 2000 to prevent getting stuck
            wall_B=0.1,       # Increased range for earlier obstacle detection
            # Enhanced behavior parameters
            enable_waiting=self.enable_enhanced_behaviors,
            enable_perturbation=self.enable_enhanced_behaviors,
            enable_panic=self.enable_enhanced_behaviors,
            waiting_density_threshold=0.8,
            perturbation_sigma=0.1,
            panic_density_threshold=1.5,
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

        - 地铁站模型 (is_metro_model=True): 8维 (3出口)
        - 旧版模型 (is_metro_model=False): 6维 (2出口)
        """
        # 计算各出口附近的密度和拥堵度
        exit_densities = []
        exit_congestions = []

        for exit_info in self.exits:
            density, congestion = self._compute_exit_metrics(exit_info['position'])
            exit_densities.append(density)
            exit_congestions.append(congestion)

        # 剩余人数比例
        remaining_ratio = len(self.model.pedestrians) / max(self.n_pedestrians, 1)

        # 时间比例
        time_ratio = min(self.step_count / 1000, 1.0)

        if self.is_metro_model:
            # 8维观测: 3出口密度 + 3出口拥堵度 + 剩余比例 + 时间比例
            obs = np.array([
                exit_densities[0], exit_densities[1], exit_densities[2],
                exit_congestions[0], exit_congestions[1], exit_congestions[2],
                remaining_ratio, time_ratio
            ], dtype=np.float32)
        else:
            # 6维观测 (兼容旧版2出口模型): 2出口密度 + 2出口拥堵度 + 剩余比例 + 时间比例
            obs = np.array([
                exit_densities[0], exit_densities[1],
                exit_congestions[0], exit_congestions[1],
                remaining_ratio, time_ratio
            ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

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
        """应用PPO引导策略

        - 地铁站模型: action 0=A, 1=B, 2=C (直接映射)
        - 旧版模型: action 0=A, 1=B (需要额外逻辑处理出口C)
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
                    prob = 0.2  # 20%的概率响应引导
                else:
                    prob = 0.08

                # 如果推荐出口不拥堵，提高响应概率
                _, congestion = self._compute_exit_metrics(target_pos)
                if congestion < 0.3:
                    prob *= 1.5

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
        """Draw info panel (English version)"""
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

        # Enhanced behavior status
        if self.enable_enhanced_behaviors:
            behavior_status = font_small.render("Enhanced Behavior: ON", True, GREEN)
        else:
            behavior_status = font_small.render("Enhanced Behavior: OFF", True, GRAY)
        screen.blit(behavior_status, (panel_x, panel_y + 85))

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
            "R: Restart",
            "ESC: Exit",
        ]

        for i, line in enumerate(lines):
            text = font_small.render(line, True, BLACK)
            screen.blit(text, (panel_x, panel_y + 105 + i * 16))

        # Legend
        legend_y = panel_y + 105 + len(lines) * 16 + 10

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
            ]
        else:
            # Speed color legend
            legends = [
                (BLUE, "Normal Speed"),
                (ORANGE, "Slow"),
                (RED, "Congested"),
                (PURPLE, "PPO Recommended"),
            ]

        for i, (color, label) in enumerate(legends):
            pygame.draw.circle(screen, color, (panel_x + 10, legend_y + i * 18 + 6), 5)
            text = font_small.render(label, True, BLACK)
            screen.blit(text, (panel_x + 25, legend_y + i * 18))

    def update(self):
        """更新模拟"""
        if self.paused:
            return

        # PPO引导（每隔一定步数更新）
        if self.ppo_model is not None and self.use_ppo:
            if self.step_count % self.ppo_update_interval == 0:
                self.apply_ppo_guidance()

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

        self.step_count += 1

    def run(self):
        """Run visualization"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Metro Evacuation - SFM + PPO Guidance")
        clock = pygame.time.Clock()

        # Use Arial font to avoid encoding issues
        font = pygame.font.SysFont('Arial', 18)
        font_small = pygame.font.SysFont('Arial', 14)

        # Load PPO model and GBM model
        self.load_ppo_model()
        self.load_gbm_predictor()
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

        pygame.quit()


def main():
    print("=" * 60)
    print("Metro Station Evacuation Simulation - Enhanced")
    print("Social Force Model + PPO Guidance + Pedestrian Types")
    print("=" * 60)
    print("\nModels:")
    print("  - Social Force Model (SFM): Pedestrian dynamics (Helbing 1995)")
    print("  - PPO Reinforcement Learning: Dynamic exit guidance")
    print("  - GBM Behavior Predictor: Based on ETH/UCY dataset")
    print("  - Pedestrian Types: Normal(70%), Elderly(15%), Child(10%), Impatient(5%)")
    print("  - Enhanced Behaviors: Waiting, Hesitation, Panic")
    print("\nLiterature:")
    print("  - Helbing 1995: Desired speed 1.34 m/s")
    print("  - Weidmann 1993: Elderly speed 0.9 m/s")
    print("  - Fruin 1971: Child speed 0.7 m/s")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  P     - Toggle PPO guidance")
    print("  T     - Toggle color mode (by type/by speed)")
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
