"""
Pygame 实时可视化人群疏散
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pygame
from sfm.social_force import SocialForceModel, Pedestrian


# 颜色定义
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (128, 128, 128)
BLUE = (70, 130, 180)
RED = (220, 80, 80)
ORANGE = (255, 165, 0)
GREEN = (60, 179, 113)
LIGHT_GREEN = (144, 238, 144)
DARK_BLUE = (25, 25, 112)
BG_COLOR = (245, 245, 245)


class PygameVisualizer:
    """Pygame 疏散可视化器"""

    def __init__(
        self,
        n_pedestrians: int = 60,
        scene_size: tuple = (30.0, 20.0),
        n_exits: int = 2,
        scale: float = 30.0,  # 像素/米
        dt: float = 0.05
    ):
        self.n_pedestrians = n_pedestrians
        self.scene_width, self.scene_height = scene_size
        self.n_exits = n_exits
        self.scale = scale
        self.dt = dt

        # 窗口大小
        self.window_width = int(self.scene_width * scale) + 150  # 额外空间显示信息
        self.window_height = int(self.scene_height * scale) + 50

        # 出口
        self.exits = []
        for i in range(n_exits):
            y_pos = (i + 1) * self.scene_height / (n_exits + 1)
            self.exits.append({
                'position': np.array([self.scene_width, y_pos]),
                'width': 2.5
            })

        # 障碍物（柱子）
        self.pillars = [
            np.array([15, 7]),
            np.array([15, 13]),
        ]

        self.model = None
        self.evacuated_count = 0
        self.step_count = 0
        self.running = True
        self.paused = False

    def world_to_screen(self, pos):
        """世界坐标转屏幕坐标"""
        x = int(pos[0] * self.scale) + 50
        y = int((self.scene_height - pos[1]) * self.scale) + 25
        return (x, y)

    def setup_model(self):
        """初始化社会力模型"""
        self.model = SocialForceModel(tau=0.5, A=2000.0, B=0.08)

        # 添加墙壁
        self.model.add_obstacle(
            np.array([0, self.scene_height]),
            np.array([self.scene_width, self.scene_height])
        )
        self.model.add_obstacle(np.array([0, 0]), np.array([self.scene_width, 0]))
        self.model.add_obstacle(np.array([0, 0]), np.array([0, self.scene_height]))

        # 右墙（带出口）
        prev_y = 0
        for exit_info in sorted(self.exits, key=lambda e: e['position'][1]):
            exit_y = exit_info['position'][1]
            half_w = exit_info['width'] / 2
            if exit_y - half_w > prev_y:
                self.model.add_obstacle(
                    np.array([self.scene_width, prev_y]),
                    np.array([self.scene_width, exit_y - half_w])
                )
            prev_y = exit_y + half_w
        if prev_y < self.scene_height:
            self.model.add_obstacle(
                np.array([self.scene_width, prev_y]),
                np.array([self.scene_width, self.scene_height])
            )

        # 柱子障碍
        for pos in self.pillars:
            size = 1.0
            self.model.add_obstacle(pos - np.array([size, 0]), pos + np.array([size, 0]))
            self.model.add_obstacle(pos - np.array([0, size]), pos + np.array([0, size]))

        # 添加行人
        np.random.seed(42)
        for i in range(self.n_pedestrians):
            position = np.array([
                np.random.uniform(2, self.scene_width * 0.4),
                np.random.uniform(2, self.scene_height - 2)
            ])

            # 选择最近的出口
            min_dist = float('inf')
            target = self.exits[0]['position']
            for exit_info in self.exits:
                dist = np.linalg.norm(position - exit_info['position'])
                if dist < min_dist:
                    min_dist = dist
                    target = exit_info['position']

            ped = Pedestrian(
                id=i,
                position=position,
                velocity=np.zeros(2),
                target=target.copy(),
                desired_speed=np.random.uniform(1.0, 1.5)
            )
            self.model.add_pedestrian(ped)

        self.evacuated_count = 0
        self.step_count = 0

    def draw_scene(self, screen, font):
        """绘制场景"""
        screen.fill(BG_COLOR)

        # 绘制出口区域
        for i, exit_info in enumerate(self.exits):
            pos = exit_info['position']
            width = exit_info['width']
            top_left = self.world_to_screen(np.array([self.scene_width, pos[1] + width/2]))
            rect_width = int(1.5 * self.scale)
            rect_height = int(width * self.scale)
            pygame.draw.rect(screen, LIGHT_GREEN,
                           (top_left[0], top_left[1], rect_width, rect_height))

            # 出口标签
            label = font.render(f"Exit {i+1}", True, GREEN)
            screen.blit(label, (top_left[0] + 5, top_left[1] + rect_height//2 - 10))

        # 绘制墙壁
        wall_points = [
            (self.world_to_screen(np.array([0, 0])),
             self.world_to_screen(np.array([self.scene_width, 0]))),
            (self.world_to_screen(np.array([0, self.scene_height])),
             self.world_to_screen(np.array([self.scene_width, self.scene_height]))),
            (self.world_to_screen(np.array([0, 0])),
             self.world_to_screen(np.array([0, self.scene_height]))),
        ]
        for start, end in wall_points:
            pygame.draw.line(screen, BLACK, start, end, 4)

        # 右墙（带出口间隙）
        prev_y = 0
        for exit_info in sorted(self.exits, key=lambda e: e['position'][1]):
            exit_y = exit_info['position'][1]
            half_w = exit_info['width'] / 2
            if exit_y - half_w > prev_y:
                start = self.world_to_screen(np.array([self.scene_width, prev_y]))
                end = self.world_to_screen(np.array([self.scene_width, exit_y - half_w]))
                pygame.draw.line(screen, BLACK, start, end, 4)
            prev_y = exit_y + half_w
        if prev_y < self.scene_height:
            start = self.world_to_screen(np.array([self.scene_width, prev_y]))
            end = self.world_to_screen(np.array([self.scene_width, self.scene_height]))
            pygame.draw.line(screen, BLACK, start, end, 4)

        # 绘制柱子
        for pos in self.pillars:
            screen_pos = self.world_to_screen(pos)
            size = int(1.5 * self.scale)
            rect = pygame.Rect(screen_pos[0] - size//2, screen_pos[1] - size//2, size, size)
            pygame.draw.rect(screen, GRAY, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)

        # 绘制行人
        for ped in self.model.pedestrians:
            screen_pos = self.world_to_screen(ped.position)
            radius = int(0.3 * self.scale)

            # 根据速度确定颜色
            speed = ped.speed
            if speed < 0.5:
                color = RED
            elif speed < 1.0:
                color = ORANGE
            else:
                color = BLUE

            pygame.draw.circle(screen, color, screen_pos, radius)
            pygame.draw.circle(screen, DARK_BLUE, screen_pos, radius, 1)

        # 绘制信息面板
        self.draw_info_panel(screen, font)

    def draw_info_panel(self, screen, font):
        """绘制信息面板"""
        panel_x = int(self.scene_width * self.scale) + 60
        panel_y = 30

        time_elapsed = self.step_count * self.dt * 3
        remaining = len(self.model.pedestrians)

        lines = [
            "=== Evacuation ===",
            f"Time: {time_elapsed:.1f}s",
            f"Evacuated: {self.evacuated_count}",
            f"Remaining: {remaining}",
            f"Total: {self.n_pedestrians}",
            "",
            "=== Controls ===",
            "SPACE: Pause",
            "R: Restart",
            "ESC: Quit",
            "",
            "=== Legend ===",
        ]

        for i, line in enumerate(lines):
            text = font.render(line, True, BLACK)
            screen.blit(text, (panel_x, panel_y + i * 22))

        # 图例
        legend_y = panel_y + len(lines) * 22
        legends = [
            (BLUE, "Normal"),
            (ORANGE, "Slow"),
            (RED, "Congested"),
        ]
        for i, (color, label) in enumerate(legends):
            pygame.draw.circle(screen, color, (panel_x + 10, legend_y + i * 25 + 10), 8)
            text = font.render(label, True, BLACK)
            screen.blit(text, (panel_x + 25, legend_y + i * 25))

    def update(self):
        """更新模拟"""
        if self.paused:
            return

        # 运行物理模拟
        for _ in range(3):
            self.model.step(self.dt)

        # 检查疏散
        to_remove = []
        for ped in self.model.pedestrians:
            for exit_info in self.exits:
                dist = np.linalg.norm(ped.position - exit_info['position'])
                if dist < exit_info['width']:
                    to_remove.append(ped)
                    self.evacuated_count += 1
                    break

        for ped in to_remove:
            self.model.pedestrians.remove(ped)

        self.step_count += 1

    def run(self):
        """运行可视化"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Crowd Evacuation Simulation - Social Force Model")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('Arial', 16)

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

            self.update()
            self.draw_scene(screen, font)
            pygame.display.flip()
            clock.tick(60)

            # 全部疏散后暂停
            if len(self.model.pedestrians) == 0 and not self.paused:
                self.paused = True
                print(f"\n疏散完成! 用时: {self.step_count * self.dt * 3:.1f}秒")

        pygame.quit()


def main():
    print("=" * 50)
    print("Pygame 人群疏散可视化")
    print("=" * 50)
    print("\n控制:")
    print("  SPACE - 暂停/继续")
    print("  R     - 重新开始")
    print("  ESC   - 退出")
    print()

    visualizer = PygameVisualizer(
        n_pedestrians=60,
        scene_size=(30.0, 20.0),
        n_exits=2,
        scale=30.0,
        dt=0.05
    )
    visualizer.run()


if __name__ == "__main__":
    main()
