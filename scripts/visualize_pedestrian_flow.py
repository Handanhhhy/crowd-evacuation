#!/usr/bin/env python
"""
Pedestrian Evacuation Simulation Visualization (Pygame)

Run LargeStationEnv in real-time with Pygame visualization.
Supports 5 methods from the research framework (see docs/new_station_plan.md 6.1).

Usage:
    # Baseline (random action)
    python scripts/visualize_pedestrian_flow.py --flow medium --method baseline

    # SFM + Density Prediction
    python scripts/visualize_pedestrian_flow.py --method prediction

    # SFM + Dynamic Routing
    python scripts/visualize_pedestrian_flow.py --method routing

    # SFM + Prediction + Routing (Full method)
    python scripts/visualize_pedestrian_flow.py --method full

    # SFM + PPO Guidance
    python scripts/visualize_pedestrian_flow.py --method ppo
    python scripts/visualize_pedestrian_flow.py --model outputs/models/ppo_large_station_small.zip

5 Methods:
    baseline   - Baseline SFM (random action)
    prediction - SFM + Density Prediction
    routing    - SFM + Dynamic Routing (load balancing)
    full       - SFM + Prediction + Routing (complete method)
    ppo        - SFM + PPO Guidance (RL-based)
"""

import sys
from pathlib import Path
import numpy as np
import pygame
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from simulation.large_station_env import LargeStationEnv
from sfm.social_force import PedestrianType

# ========== 视觉风格配置 (参考 station.png) ==========
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

COLORS = {
    'bg': (248, 248, 248),           # 背景浅灰
    'wall': (50, 50, 50),            # 墙壁深灰
    'floor': (255, 255, 255),        # 地板白色
    'escalator_bg': (220, 220, 220), # 扶梯背景
    'stairs': (65, 105, 225),        # 步梯蓝色 (RoyalBlue)
    'escalator': (192, 192, 192),    # 扶梯灰色
    'gate': (0, 0, 0),               # 闸机黑色
    'exit_arrow': (0, 128, 0),       # 出口绿色
    'text': (0, 0, 0),               # 文字
}

# 行人颜色
PED_COLORS = {
    PedestrianType.NORMAL: (30, 144, 255),          # 蓝
    PedestrianType.ELDERLY: (128, 0, 128),          # 紫
    PedestrianType.CHILD: (255, 165, 0),            # 橙
    PedestrianType.IMPATIENT: (220, 20, 60),        # 红
    PedestrianType.WITH_SMALL_BAG: (70, 130, 180),  # 钢蓝
    PedestrianType.WITH_LUGGAGE: (46, 139, 87),     # 海绿
    PedestrianType.WITH_LARGE_LUGGAGE: (139, 69, 19) # 棕
}

class StationVisualizer:
    def __init__(self, flow_level="medium", scale=8.0, model_path=None, method="baseline"):
        self.scale = scale
        self.margin = 50
        self.method = method
        self.model = None

        # Initialize environment
        print(f"Initializing environment (flow: {flow_level}, method: {method})...")
        self.env = LargeStationEnv(
            flow_level=flow_level,
            use_gpu_sfm=True,  # 尝试使用GPU
            render_mode=None
        )
        self.env.reset()

        # 场景尺寸（必须在加载模型前设置）
        self.scene_w, self.scene_h = self.env.scene_width, self.env.scene_height

        # 窗口尺寸
        self.width = int(self.scene_w * scale) + self.margin * 2
        self.height = int(self.scene_h * scale) + self.margin * 2

        # Load PPO model (if specified)
        if model_path or method == "ppo":
            self._load_model(model_path)
            # Fallback to 'full' method if model not found
            if self.model is None:
                print("Fallback to 'full' method (SFM+Prediction+Routing)")
                self.method = "full"

        # 状态
        self.running = True
        self.paused = False
        self.step_count = 0

        # 统计信息
        self.total_evacuated = 0
        self.max_density = 0
        self.evacuation_start_time = time.time()

        # 缓存墙壁数据（转为屏幕坐标）
        self.walls_screen = []
        walls = []
        if hasattr(self.env.sfm, 'walls'):
            walls = self.env.sfm.walls
        elif hasattr(self.env.sfm, 'obstacles'):
            walls = self.env.sfm.obstacles

        for wall in walls:
            if hasattr(wall, 'cpu'): wall = wall.cpu().numpy()
            start = self.world_to_screen(wall[0])
            end = self.world_to_screen(wall[1])
            self.walls_screen.append((start, end))

    def _load_model(self, model_path=None):
        """Load PPO model"""
        if model_path is None:
            # Default model path
            model_path = project_root / "outputs/models/ppo_large_station_small.zip"
        else:
            model_path = Path(model_path)

        if model_path.exists():
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(str(model_path))
                print(f"Loaded PPO model: {model_path}")
            except Exception as e:
                print(f"Warning: Cannot load PPO model ({e})")
                self.model = None
        else:
            print(f"Note: Model file not found ({model_path})")

    def _get_action(self, obs):
        """根据方案选择动作"""
        if self.method == "ppo" and self.model is not None:
            # PPO模型预测
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        elif self.method in ["routing", "full", "prediction"]:
            # 动态分流：选择负载最小的出口
            # prediction和full方案也使用分流（密度预测集成在环境中）
            return self._routing_action()
        else:
            # 基线：随机动作
            return self.env.action_space.sample()

    def _routing_action(self):
        """动态分流策略：选择负载最小的出口"""
        n_exits = self.env.action_space.n

        if hasattr(self.env, 'evacuated_by_exit') and self.env.evacuated_by_exit:
            loads = np.array([self.env.evacuated_by_exit.get(i, 0) for i in range(n_exits)])
        else:
            loads = np.zeros(n_exits)

        # 选择负载最小的出口（添加小扰动避免总选同一个）
        scores = -loads + np.random.uniform(0, 0.1, n_exits)
        return int(np.argmax(scores))

    def world_to_screen(self, pos):
        """世界坐标(米) -> 屏幕坐标(像素)"""
        # Pygame坐标系: 原点在左上，y向下
        # 仿真坐标系: 通常y向上 (需要翻转y)
        x = self.margin + int(pos[0] * self.scale)
        y = self.height - self.margin - int(pos[1] * self.scale)
        return (x, y)

    def draw_structure(self, screen, font):
        """绘制车站静态结构"""
        # 1. 绘制地板区域 (T形)
        pts = [
            (0, 0), (20, 0), (20, 10), (150, 10),
            (150, 70), (20, 70), (20, 80), (0, 80)
        ]
        poly_pts = [self.world_to_screen(p) for p in pts]
        pygame.draw.polygon(screen, COLORS['floor'], poly_pts)

        # 2. 绘制直升电梯 (完全禁行障碍物)
        if hasattr(self.env, 'elevator'):
            elev = self.env.elevator
            ex, ey = elev["position"]
            ew, eh = elev["size"]
            tl = self.world_to_screen((ex - ew/2, ey + eh/2))
            br = self.world_to_screen((ex + ew/2, ey - eh/2))
            rect = pygame.Rect(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])
            pygame.draw.rect(screen, (255, 200, 200), rect)  # 浅红色背景

            # 绘制X斜线表示完全禁行
            line_color = (180, 0, 0)
            line_spacing = 10
            # 正斜线
            for i in range(-rect.height, rect.width, line_spacing):
                start_x = rect.left + max(0, i)
                start_y = rect.top + max(0, -i)
                end_x = rect.left + min(rect.width, i + rect.height)
                end_y = rect.top + min(rect.height, rect.height - i)
                if start_x < rect.right and end_x > rect.left:
                    pygame.draw.line(screen, line_color, (start_x, start_y), (end_x, end_y), 1)
            # 反斜线
            for i in range(-rect.height, rect.width, line_spacing):
                start_x = rect.right - max(0, i)
                start_y = rect.top + max(0, -i)
                end_x = rect.right - min(rect.width, i + rect.height)
                end_y = rect.top + min(rect.height, rect.height - i)
                if start_x > rect.left and end_x < rect.right:
                    pygame.draw.line(screen, line_color, (start_x, start_y), (end_x, end_y), 1)

            pygame.draw.rect(screen, (139, 0, 0), rect, 2)  # 深红边框

        # 3. 绘制扶梯/楼梯区域
        # 步梯：整体蓝色，无禁行区
        # 扶梯：运送区（灰色）+ 禁行区（红X）
        for esc in self.env.escalators:
            w, h = esc.size
            x, y = esc.position
            x1, y1 = x - w/2, y - h/2
            exit_edge = getattr(esc, 'exit_edge', 'down')
            is_stairs = 'stair' in esc.id

            # 整体矩形
            tl = self.world_to_screen((x1, y1 + h))
            br = self.world_to_screen((x1 + w, y1))
            full_rect = pygame.Rect(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])

            if is_stairs:
                # 步梯：整体蓝色，无禁行区
                pygame.draw.rect(screen, COLORS['stairs'], full_rect)
                pygame.draw.rect(screen, BLACK, full_rect, 2)

                # 步梯：出口边缘绿色
                exit_color = (0, 180, 0)
                if exit_edge == "left":
                    pygame.draw.line(screen, exit_color,
                                    (full_rect.left, full_rect.top), (full_rect.left, full_rect.bottom), 5)
                elif exit_edge == "right":
                    pygame.draw.line(screen, exit_color,
                                    (full_rect.right, full_rect.top), (full_rect.right, full_rect.bottom), 5)
                elif exit_edge == "down":
                    pygame.draw.line(screen, exit_color,
                                    (full_rect.left, full_rect.bottom), (full_rect.right, full_rect.bottom), 5)
                else:  # up
                    pygame.draw.line(screen, exit_color,
                                    (full_rect.left, full_rect.top), (full_rect.right, full_rect.top), 5)

                # 步梯：绿色箭头指向出口
                arrow_color = (0, 180, 0)
                cx, cy = full_rect.centerx, full_rect.centery
                arrow_size = min(full_rect.width, full_rect.height) // 4

                if exit_edge == "left":
                    pygame.draw.line(screen, arrow_color, (cx + arrow_size, cy), (cx - arrow_size, cy), 3)
                    pygame.draw.line(screen, arrow_color, (cx - arrow_size, cy), (cx - arrow_size//2, cy - arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx - arrow_size, cy), (cx - arrow_size//2, cy + arrow_size//2), 3)
                elif exit_edge == "right":
                    pygame.draw.line(screen, arrow_color, (cx - arrow_size, cy), (cx + arrow_size, cy), 3)
                    pygame.draw.line(screen, arrow_color, (cx + arrow_size, cy), (cx + arrow_size//2, cy - arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx + arrow_size, cy), (cx + arrow_size//2, cy + arrow_size//2), 3)
                elif exit_edge == "down":
                    pygame.draw.line(screen, arrow_color, (cx, cy - arrow_size), (cx, cy + arrow_size), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy + arrow_size), (cx - arrow_size//2, cy + arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy + arrow_size), (cx + arrow_size//2, cy + arrow_size//2), 3)
                else:  # up
                    pygame.draw.line(screen, arrow_color, (cx, cy + arrow_size), (cx, cy - arrow_size), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy - arrow_size), (cx - arrow_size//2, cy - arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy - arrow_size), (cx + arrow_size//2, cy - arrow_size//2), 3)

            else:
                # 扶梯：分为运送区和禁行区
                is_horizontal = (w > h)
                transport_ratio = 0.5

                if is_horizontal:
                    # 横向扶梯：水平分隔线
                    split_y = full_rect.top + int(full_rect.height * transport_ratio)
                    transport_rect = pygame.Rect(full_rect.left, full_rect.top,
                                                 full_rect.width, split_y - full_rect.top)
                    nogo_rect = pygame.Rect(full_rect.left, split_y,
                                            full_rect.width, full_rect.bottom - split_y)
                    split_line = ((full_rect.left, split_y), (full_rect.right, split_y))
                else:
                    # 纵向扶梯：垂直分隔线
                    split_x = full_rect.left + int(full_rect.width * transport_ratio)
                    transport_rect = pygame.Rect(full_rect.left, full_rect.top,
                                                 split_x - full_rect.left, full_rect.height)
                    nogo_rect = pygame.Rect(split_x, full_rect.top,
                                            full_rect.right - split_x, full_rect.height)
                    split_line = ((split_x, full_rect.top), (split_x, full_rect.bottom))

                # 绘制运送区（灰色）
                pygame.draw.rect(screen, (180, 180, 180), transport_rect)

                # 绘制禁行区（白色背景 + 红色X斜线）
                pygame.draw.rect(screen, WHITE, nogo_rect)

                # 使用clip确保斜线不溢出
                line_color = (200, 60, 60)
                line_spacing = 8
                screen.set_clip(nogo_rect)

                # 正斜线 + 反斜线 = X
                for i in range(-nogo_rect.height, nogo_rect.width + nogo_rect.height, line_spacing):
                    pygame.draw.line(screen, line_color,
                                    (nogo_rect.left + i, nogo_rect.top),
                                    (nogo_rect.left + i - nogo_rect.height, nogo_rect.bottom), 1)
                    pygame.draw.line(screen, line_color,
                                    (nogo_rect.left + i, nogo_rect.top),
                                    (nogo_rect.left + i + nogo_rect.height, nogo_rect.bottom), 1)

                screen.set_clip(None)

                # 绘制边框（黑色）
                pygame.draw.rect(screen, BLACK, full_rect, 2)
                # 绘制分隔线
                pygame.draw.line(screen, BLACK, split_line[0], split_line[1], 2)

                # 扶梯：出口边也分为两部分
                # - 运送区出口：绿色（可出人）
                # - 禁行区出口：黑色（不能出人）
                exit_color = (0, 180, 0)

                if is_horizontal:
                    # 横向扶梯：出口边分上下两部分
                    if exit_edge == "left":
                        # 左边上半（运送区）绿色，左边下半（禁行区）黑色
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.left, full_rect.top), (full_rect.left, split_y), 5)
                        pygame.draw.line(screen, BLACK,
                                        (full_rect.left, split_y), (full_rect.left, full_rect.bottom), 5)
                    elif exit_edge == "right":
                        # 右边上半（运送区）绿色，右边下半（禁行区）黑色
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.right, full_rect.top), (full_rect.right, split_y), 5)
                        pygame.draw.line(screen, BLACK,
                                        (full_rect.right, split_y), (full_rect.right, full_rect.bottom), 5)
                    elif exit_edge == "down":
                        # 下边全绿（运送区在上，但出口在下，整条出口都是运送区的出口）
                        # 实际上横向扶梯不应该有down/up出口，但为完整性保留
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.left, full_rect.bottom), (full_rect.right, full_rect.bottom), 5)
                    else:  # up
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.left, full_rect.top), (full_rect.right, full_rect.top), 5)
                else:
                    # 纵向扶梯：出口边分左右两部分
                    if exit_edge == "down":
                        # 下边左半（运送区）绿色，下边右半（禁行区）黑色
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.left, full_rect.bottom), (split_x, full_rect.bottom), 5)
                        pygame.draw.line(screen, BLACK,
                                        (split_x, full_rect.bottom), (full_rect.right, full_rect.bottom), 5)
                    elif exit_edge == "up":
                        # 上边左半（运送区）绿色，上边右半（禁行区）黑色
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.left, full_rect.top), (split_x, full_rect.top), 5)
                        pygame.draw.line(screen, BLACK,
                                        (split_x, full_rect.top), (full_rect.right, full_rect.top), 5)
                    elif exit_edge == "left":
                        # 纵向扶梯不应该有left/right出口，但为完整性保留
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.left, full_rect.top), (full_rect.left, full_rect.bottom), 5)
                    else:  # right
                        pygame.draw.line(screen, exit_color,
                                        (full_rect.right, full_rect.top), (full_rect.right, full_rect.bottom), 5)

                # 扶梯：绿色箭头在运送区中央
                arrow_color = (0, 180, 0)
                cx, cy = transport_rect.centerx, transport_rect.centery
                arrow_size = min(transport_rect.width, transport_rect.height) // 3

                if exit_edge == "left":
                    pygame.draw.line(screen, arrow_color, (cx + arrow_size, cy), (cx - arrow_size, cy), 3)
                    pygame.draw.line(screen, arrow_color, (cx - arrow_size, cy), (cx - arrow_size//2, cy - arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx - arrow_size, cy), (cx - arrow_size//2, cy + arrow_size//2), 3)
                elif exit_edge == "right":
                    pygame.draw.line(screen, arrow_color, (cx - arrow_size, cy), (cx + arrow_size, cy), 3)
                    pygame.draw.line(screen, arrow_color, (cx + arrow_size, cy), (cx + arrow_size//2, cy - arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx + arrow_size, cy), (cx + arrow_size//2, cy + arrow_size//2), 3)
                elif exit_edge == "down":
                    pygame.draw.line(screen, arrow_color, (cx, cy - arrow_size), (cx, cy + arrow_size), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy + arrow_size), (cx - arrow_size//2, cy + arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy + arrow_size), (cx + arrow_size//2, cy + arrow_size//2), 3)
                else:  # up
                    pygame.draw.line(screen, arrow_color, (cx, cy + arrow_size), (cx, cy - arrow_size), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy - arrow_size), (cx - arrow_size//2, cy - arrow_size//2), 3)
                    pygame.draw.line(screen, arrow_color, (cx, cy - arrow_size), (cx + arrow_size//2, cy - arrow_size//2), 3)

        # 4. 绘制墙壁/围栏
        for start, end in self.walls_screen:
            pygame.draw.line(screen, COLORS['wall'], start, end, 3)

        # 4.1 绘制T形边界墙（黑色粗线，与闸机相连）
        # 闸机位置: b(45-60), c(75-90), d(105-120) 在Y=70
        #          e(45-60), f(75-90), g(105-120) 在Y=10
        #          a(30-50) 在X=0, zi(30-50) 在X=150
        boundary_walls = [
            # 左侧走廊
            ((0, 0), (0, 30)),       # 左墙下段
            ((0, 50), (0, 80)),      # 左墙上段 (闸机a在30-50)
            ((0, 80), (20, 80)),     # 左廊顶边
            ((0, 0), (20, 0)),       # 左廊底边
            # 主厅上边 (Y=70, 闸机b,c,d间隔)
            ((20, 70), (45, 70)),
            ((60, 70), (75, 70)),
            ((90, 70), (105, 70)),
            ((120, 70), (150, 70)),
            # 主厅下边 (Y=10, 闸机e,f,g间隔)
            ((20, 10), (45, 10)),
            ((60, 10), (75, 10)),
            ((90, 10), (105, 10)),
            ((120, 10), (150, 10)),
            # 右墙 (闸机zi在30-50)
            ((150, 10), (150, 30)),
            ((150, 50), (150, 70)),
            # 连接处
            ((20, 70), (20, 80)),    # 左廊与主厅上连接
            ((20, 0), (20, 10)),     # 左廊与主厅下连接
        ]
        for start, end in boundary_walls:
            s = self.world_to_screen(start)
            e = self.world_to_screen(end)
            pygame.draw.line(screen, BLACK, s, e, 3)

        # 5. 绘制闸机/出口
        for exit in self.env.exits:
            pos = self.world_to_screen(exit.position)
            length = int(exit.width * self.scale)

            if exit.direction in ['up', 'down']:
                start = (pos[0] - length//2, pos[1])
                end = (pos[0] + length//2, pos[1])
            else:
                start = (pos[0], pos[1] - length//2)
                end = (pos[0], pos[1] + length//2)

            pygame.draw.line(screen, COLORS['exit_arrow'], start, end, 5)

            # 标签
            label = exit.name.replace('闸机', '')
            txt = font.render(label, True, COLORS['text'])
            screen.blit(txt, (pos[0] - 10, pos[1] - 20))

    def draw_pedestrians(self, screen):
        """绘制行人"""
        # 获取行人数据
        if hasattr(self.env.sfm, 'pedestrians'): # CPU
            peds = self.env.sfm.pedestrians
            for ped in peds:
                pos = self.world_to_screen(ped.position)
                color = PED_COLORS.get(ped.ped_type, PED_COLORS[PedestrianType.NORMAL])
                pygame.draw.circle(screen, color, pos, 3)
        
        elif hasattr(self.env.sfm, '_positions_tensor'): # GPU
            # GPU模式下数据在tensor里，且类型信息可能在别处
            # 为简单起见，统一用一种颜色，或尝试获取类型
            positions = self.env.sfm._positions_tensor.cpu().numpy()
            
            # 批量绘制
            for i in range(len(positions)):
                pos = self.world_to_screen(positions[i])
                pygame.draw.circle(screen, PED_COLORS[PedestrianType.NORMAL], pos, 3)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Station Evacuation Simulation")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 16) # 也可以尝试加载中文字体
        
        while self.running:
            # 1. 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self.running = False
                    elif event.key == pygame.K_SPACE: self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        self.step_count = 0
                        self.total_evacuated = 0
                        self.max_density = 0
                        self.evacuation_start_time = time.time()
                        print("Environment reset")
                    # Number keys 1-5 to switch methods
                    elif event.key == pygame.K_1:
                        self.method = "baseline"
                        print("Switched to: Baseline SFM")
                    elif event.key == pygame.K_2:
                        self.method = "prediction"
                        print("Switched to: SFM+Prediction")
                    elif event.key == pygame.K_3:
                        self.method = "routing"
                        print("Switched to: SFM+Routing")
                    elif event.key == pygame.K_4:
                        self.method = "full"
                        print("Switched to: SFM+Pred+Route (Full)")
                    elif event.key == pygame.K_5:
                        if self.model is not None:
                            self.method = "ppo"
                            print("Switched to: SFM+PPO")
                        else:
                            print("PPO model not loaded, cannot switch to PPO method")
            
            # 2. 物理更新
            if not self.paused:
                # 获取观测并选择动作
                obs = self.env._get_observation()
                action = self._get_action(obs)
                _, reward, done, truncated, info = self.env.step(action)
                self.step_count += 1

                # 更新统计
                self.total_evacuated = info.get('evacuated', 0)
                self.max_density = max(self.max_density, info.get('max_density', 0))

                # 检查是否完成
                if done or truncated:
                    elapsed = time.time() - self.evacuation_start_time
                    evac_rate = self.total_evacuated / self.env.n_pedestrians * 100
                    print(f"\nEvacuation complete! Rate: {evac_rate:.1f}%, Time: {elapsed:.1f}s, Max density: {self.max_density:.2f}/m2")

            # 3. 渲染
            screen.fill(COLORS['bg'])
            self.draw_structure(screen, font)
            self.draw_pedestrians(screen)

            # 4. UI信息
            if hasattr(self.env.sfm, 'pedestrians'):
                n_peds = len(self.env.sfm.pedestrians)
            else:
                n_peds = self.env.sfm.get_active_count() if hasattr(self.env.sfm, 'get_active_count') else 0

            evac_rate = self.total_evacuated / self.env.n_pedestrians * 100 if self.env.n_pedestrians > 0 else 0
            sim_time = self.step_count * self.env.dt

            # Method names (English to avoid font issues)
            method_names = {
                "baseline": "Baseline SFM",
                "prediction": "SFM+Prediction",
                "routing": "SFM+Routing",
                "full": "SFM+Pred+Route",
                "ppo": "SFM+PPO"
            }
            method_name = method_names.get(self.method, self.method)

            # Line 1: Basic info
            line1 = f"Method: {method_name} | Time: {sim_time:.1f}s | FPS: {clock.get_fps():.1f}"
            info_surf = font.render(line1, True, COLORS['text'])
            screen.blit(info_surf, (10, 10))

            # Line 2: Evacuation info
            line2 = f"Remaining: {n_peds} | Evacuated: {self.total_evacuated}/{self.env.n_pedestrians} ({evac_rate:.1f}%) | Max Density: {self.max_density:.2f}/m2"
            info_surf2 = font.render(line2, True, COLORS['text'])
            screen.blit(info_surf2, (10, 30))

            # Line 3: Method selector (highlight current in RED)
            method_keys = [("1", "baseline"), ("2", "prediction"), ("3", "routing"), ("4", "full"), ("5", "ppo")]
            x_offset = 10
            for key, m in method_keys:
                if m == self.method:
                    # Selected: RED + UPPERCASE
                    text = f"[{key}]{m.upper()}"
                    color = (220, 20, 60)  # Crimson red
                else:
                    # Not selected: gray
                    text = f"[{key}]{m}"
                    color = (128, 128, 128)
                surf = font.render(text, True, color)
                screen.blit(surf, (x_offset, 50))
                x_offset += surf.get_width() + 15  # spacing between items

            # Line 4: Controls (at bottom)
            line4 = "[Space] Pause  [R] Reset  [Esc] Quit"
            info_surf4 = font.render(line4, True, (128, 128, 128))
            screen.blit(info_surf4, (10, self.height - 25))
            
            pygame.display.flip()
            clock.tick(30) # 限制30FPS
            
        pygame.quit()
        self.env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pedestrian Evacuation Simulation Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods (5 from research framework):
  baseline   - Baseline SFM with random action
  prediction - SFM + Density Prediction
  routing    - SFM + Dynamic Routing (load balancing)
  full       - SFM + Prediction + Routing (complete method)
  ppo        - SFM + PPO Guidance (requires trained model)

Examples:
  python scripts/visualize_pedestrian_flow.py --flow medium
  python scripts/visualize_pedestrian_flow.py --method full --flow large
  python scripts/visualize_pedestrian_flow.py --method ppo --model outputs/models/ppo_large_station_small.zip
        """
    )
    parser.add_argument("--flow", default="medium", choices=["small", "medium", "large"],
                        help="Flow level: small/medium/large (default: medium)")
    parser.add_argument("--model", type=str, default=None,
                        help="PPO model path (auto-enables ppo method)")
    parser.add_argument("--method", default="baseline",
                        choices=["baseline", "prediction", "routing", "full", "ppo"],
                        help="Method: baseline/prediction/routing/full/ppo (default: baseline)")
    parser.add_argument("--scale", type=float, default=8.0,
                        help="Display scale factor (default: 8.0)")
    args = parser.parse_args()

    # If model path is specified, auto-enable ppo method
    method = args.method
    if args.model:
        method = "ppo"

    viz = StationVisualizer(
        flow_level=args.flow,
        scale=args.scale,
        model_path=args.model,
        method=method
    )
    viz.run()