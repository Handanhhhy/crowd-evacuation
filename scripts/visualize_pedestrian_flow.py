#!/usr/bin/env python
"""
行人疏散仿真可视化 (Pygame)

实时运行 LargeStationEnv 并在 Pygame 中展示。
参考 examples/plot_large_station_layout.py 的视觉风格。
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
    def __init__(self, flow_level="medium", scale=8.0):
        self.scale = scale
        self.margin = 50
        
        # 初始化环境
        print(f"初始化环境 (流量: {flow_level})...")
        self.env = LargeStationEnv(
            flow_level=flow_level,
            use_gpu_sfm=True,  # 尝试使用GPU
            render_mode=None
        )
        self.env.reset()
        
        # 场景尺寸
        self.scene_w, self.scene_h = self.env.scene_width, self.env.scene_height
        
        # 窗口尺寸
        self.width = int(self.scene_w * scale) + self.margin * 2
        self.height = int(self.scene_h * scale) + self.margin * 2
        
        # 状态
        self.running = True
        self.paused = False
        self.step_count = 0
        self.fps = 0
        
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

        # 2. 绘制直升电梯 (障碍物)
        if hasattr(self.env, 'elevator'):
            elev = self.env.elevator
            ex, ey = elev["position"]
            ew, eh = elev["size"]
            tl = self.world_to_screen((ex - ew/2, ey + eh/2))
            br = self.world_to_screen((ex + ew/2, ey - eh/2))
            rect = pygame.Rect(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])
            pygame.draw.rect(screen, (255, 127, 127), rect)  # 浅红色
            pygame.draw.rect(screen, (139, 0, 0), rect, 2)   # 深红边框
            # X标记
            pygame.draw.line(screen, (139, 0, 0), rect.topleft, rect.bottomright, 2)
            pygame.draw.line(screen, (139, 0, 0), rect.topright, rect.bottomleft, 2)

        # 3. 绘制扶梯/楼梯区域（障碍物 + 出口边缘）
        for esc in self.env.escalators:
            w, h = esc.size
            x, y = esc.position
            x1, y1 = x - w/2, y - h/2

            tl = self.world_to_screen((x1, y1 + h))
            br = self.world_to_screen((x1 + w, y1))
            rect = pygame.Rect(tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])

            # 区分步梯和扶梯 (都是障碍物)
            color = COLORS['stairs'] if 'stair' in esc.id else COLORS['escalator']
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)

            # 绘制出口边缘（绿色线）
            exit_color = (0, 200, 0)  # 亮绿色
            exit_edge = getattr(esc, 'exit_edge', 'down')
            if exit_edge == "left":
                start = self.world_to_screen((x1, y1))
                end = self.world_to_screen((x1, y1 + h))
                pygame.draw.line(screen, exit_color, start, end, 4)
            elif exit_edge == "right":
                start = self.world_to_screen((x1 + w, y1))
                end = self.world_to_screen((x1 + w, y1 + h))
                pygame.draw.line(screen, exit_color, start, end, 4)
            elif exit_edge == "down":
                start = self.world_to_screen((x1, y1))
                end = self.world_to_screen((x1 + w, y1))
                pygame.draw.line(screen, exit_color, start, end, 4)
            elif exit_edge == "up":
                start = self.world_to_screen((x1, y1 + h))
                end = self.world_to_screen((x1 + w, y1 + h))
                pygame.draw.line(screen, exit_color, start, end, 4)

            # 绘制出口方向箭头
            arrow_map = {"left": "←", "right": "→", "up": "↑", "down": "↓"}
            arrow_txt = arrow_map.get(exit_edge, "↓")
            txt_surf = font.render(arrow_txt, True, WHITE)
            screen.blit(txt_surf, (rect.centerx - txt_surf.get_width()/2, rect.centery - txt_surf.get_height()/2))

        # 4. 绘制墙壁/围栏
        for start, end in self.walls_screen:
            pygame.draw.line(screen, COLORS['wall'], start, end, 3)

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
                    elif event.key == pygame.K_r: self.env.reset()
            
            # 2. 物理更新
            if not self.paused:
                # 随机动作（对于Social Force Model，动作通常由模型内部处理，这里只是驱动step）
                action = self.env.action_space.sample() 
                self.env.step(action)
                self.step_count += 1
            
            # 3. 渲染
            screen.fill(COLORS['bg'])
            self.draw_structure(screen, font)
            self.draw_pedestrians(screen)
            
            # 4. UI信息
            info = f"Step: {self.step_count} | Pedestrians: {len(self.env.sfm.pedestrians) if hasattr(self.env.sfm, 'pedestrians') else len(self.env.sfm._positions_tensor)} | FPS: {clock.get_fps():.1f}"
            info_surf = font.render(info, True, COLORS['text'])
            screen.blit(info_surf, (10, 10))
            
            pygame.display.flip()
            clock.tick(30) # 限制30FPS
            
        pygame.quit()
        self.env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", default="medium", choices=["small", "medium", "large"])
    args = parser.parse_args()
    
    viz = StationVisualizer(flow_level=args.flow)
    viz.run()