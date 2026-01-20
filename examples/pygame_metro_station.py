"""
成都东客站地铁出站口疏散仿真
基于开题报告场景建模
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
LIGHT_GRAY = (200, 200, 200)
BLUE = (70, 130, 180)
RED = (220, 80, 80)
ORANGE = (255, 165, 0)
GREEN = (60, 179, 113)
LIGHT_GREEN = (144, 238, 144)
DARK_BLUE = (25, 25, 112)
YELLOW = (255, 215, 0)
BG_COLOR = (240, 240, 235)
FLOOR_COLOR = (220, 220, 210)
GATE_COLOR = (100, 100, 100)


class MetroStationVisualizer:
    """成都东客站地铁出站口可视化"""

    def __init__(
        self,
        n_pedestrians: int = 80,
        scale: float = 12.0,
        dt: float = 0.05
    ):
        self.n_pedestrians = n_pedestrians
        self.scale = scale
        self.dt = dt

        # 场景尺寸 (米) - 模拟地铁出站口
        # 大致布局：站台区 -> 闸机区 -> 出站大厅 -> 多个出口
        self.scene_width = 60.0   # 总宽度
        self.scene_height = 40.0  # 总高度

        # 窗口大小
        self.window_width = int(self.scene_width * scale) + 200
        self.window_height = int(self.scene_height * scale) + 60

        # 区域定义
        self.platform_area = (0, 10, 15, 30)      # 站台区域 (x1, y1, x2, y2)
        self.gate_area = (15, 5, 25, 35)          # 闸机区域
        self.hall_area = (25, 0, 60, 40)          # 出站大厅

        # 闸机位置 (模拟多个闸机通道)
        self.gates = []
        gate_y_positions = [10, 15, 20, 25, 30]
        for y in gate_y_positions:
            self.gates.append({
                'position': np.array([20.0, y]),
                'width': 0.8,
                'length': 3.0
            })

        # 出口定义 (多个出口)
        self.exits = [
            {'id': 'A', 'position': np.array([60, 10]), 'width': 4.0, 'label': '出口A'},
            {'id': 'B', 'position': np.array([60, 30]), 'width': 4.0, 'label': '出口B'},
            {'id': 'C', 'position': np.array([40, 40]), 'width': 5.0, 'label': '出口C (主出口)'},
        ]

        # 柱子位置
        self.pillars = [
            np.array([30, 12]),
            np.array([30, 28]),
            np.array([45, 12]),
            np.array([45, 28]),
            np.array([35, 20]),
            np.array([50, 20]),
        ]

        # 障碍物/设施
        self.facilities = [
            {'type': 'info', 'position': np.array([32, 20]), 'size': (3, 2), 'label': '信息台'},
            {'type': 'stairs', 'position': np.array([55, 20]), 'size': (4, 8), 'label': '楼梯'},
        ]

        self.model = None
        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.step_count = 0
        self.running = True
        self.paused = False

    def world_to_screen(self, pos):
        """世界坐标转屏幕坐标"""
        x = int(pos[0] * self.scale) + 50
        y = int((self.scene_height - pos[1]) * self.scale) + 30
        return (x, y)

    def setup_model(self):
        """初始化社会力模型"""
        self.model = SocialForceModel(tau=0.5, A=2000.0, B=0.08)

        # 外墙
        # 上墙
        self.model.add_obstacle(
            np.array([0, self.scene_height]),
            np.array([35, self.scene_height])
        )
        self.model.add_obstacle(
            np.array([45, self.scene_height]),
            np.array([self.scene_width, self.scene_height])
        )

        # 下墙
        self.model.add_obstacle(
            np.array([0, 0]),
            np.array([self.scene_width, 0])
        )

        # 左墙
        self.model.add_obstacle(np.array([0, 0]), np.array([0, self.scene_height]))

        # 右墙 (带出口)
        self.model.add_obstacle(np.array([self.scene_width, 0]), np.array([self.scene_width, 8]))
        self.model.add_obstacle(np.array([self.scene_width, 12]), np.array([self.scene_width, 28]))
        self.model.add_obstacle(np.array([self.scene_width, 32]), np.array([self.scene_width, self.scene_height]))

        # 闸机区域障碍物
        # 闸机布局: 5个通道，每个通道宽1.2米，闸机设备宽2米
        gate_x = 20.0  # 闸机中心x坐标
        gate_length = 3.0  # 闸机长度
        passage_width = 1.2  # 通道宽度
        barrier_height = 1.5  # 隔板高度（闸机设备）

        # 闸机通道的y坐标: 10, 15, 20, 25, 30
        # 隔板在通道之间: 7, 12.5, 17.5, 22.5, 27.5, 33

        # 闸机隔板 (通道之间的障碍物)
        barrier_y_positions = [7, 12.5, 17.5, 22.5, 27.5, 33]
        for by in barrier_y_positions:
            # 隔板是横向的短墙
            self.model.add_obstacle(
                np.array([gate_x - gate_length/2, by]),
                np.array([gate_x + gate_length/2, by])
            )

        # 柱子
        for pos in self.pillars:
            size = 0.8
            self.model.add_obstacle(pos - np.array([size, 0]), pos + np.array([size, 0]))
            self.model.add_obstacle(pos - np.array([0, size]), pos + np.array([0, size]))

        # 设施障碍
        for facility in self.facilities:
            pos = facility['position']
            w, h = facility['size']
            # 四边
            self.model.add_obstacle(
                np.array([pos[0] - w/2, pos[1] - h/2]),
                np.array([pos[0] + w/2, pos[1] - h/2])
            )
            self.model.add_obstacle(
                np.array([pos[0] - w/2, pos[1] + h/2]),
                np.array([pos[0] + w/2, pos[1] + h/2])
            )
            self.model.add_obstacle(
                np.array([pos[0] - w/2, pos[1] - h/2]),
                np.array([pos[0] - w/2, pos[1] + h/2])
            )
            self.model.add_obstacle(
                np.array([pos[0] + w/2, pos[1] - h/2]),
                np.array([pos[0] + w/2, pos[1] + h/2])
            )

        # 添加行人 (从站台区域生成)
        np.random.seed(None)  # 随机种子
        for i in range(self.n_pedestrians):
            # 在站台区域随机生成
            position = np.array([
                np.random.uniform(2, 14),
                np.random.uniform(12, 28)
            ])

            # 智能选择出口 (基于位置)
            target = self._choose_exit(position)

            ped = Pedestrian(
                id=i,
                position=position,
                velocity=np.zeros(2),
                target=target.copy(),
                desired_speed=np.random.uniform(1.0, 1.6)
            )
            self.model.add_pedestrian(ped)

        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.step_count = 0

    def _choose_exit(self, position):
        """智能选择出口"""
        # 根据y位置倾向性选择出口
        y = position[1]

        if y < 15:
            # 下方的人倾向于出口A
            weights = [0.6, 0.2, 0.2]
        elif y > 25:
            # 上方的人倾向于出口B
            weights = [0.2, 0.6, 0.2]
        else:
            # 中间的人倾向于主出口C
            weights = [0.2, 0.2, 0.6]

        choice = np.random.choice([0, 1, 2], p=weights)
        return self.exits[choice]['position'].copy()

    def draw_scene(self, screen, font, font_small):
        """绘制场景"""
        screen.fill(BG_COLOR)

        # 绘制地面区域
        # 站台区
        platform_rect = pygame.Rect(
            self.world_to_screen(np.array([0, 30]))[0],
            self.world_to_screen(np.array([0, 30]))[1],
            int(15 * self.scale),
            int(20 * self.scale)
        )
        pygame.draw.rect(screen, (200, 180, 160), platform_rect)

        # 出站大厅
        hall_rect = pygame.Rect(
            self.world_to_screen(np.array([25, 40]))[0],
            self.world_to_screen(np.array([25, 40]))[1],
            int(35 * self.scale),
            int(40 * self.scale)
        )
        pygame.draw.rect(screen, FLOOR_COLOR, hall_rect)

        # 绘制出口区域
        for exit_info in self.exits:
            pos = exit_info['position']
            width = exit_info['width']

            if pos[0] >= self.scene_width - 0.1:  # 右侧出口
                top_left = self.world_to_screen(np.array([self.scene_width - 1, pos[1] + width/2]))
                rect_w = int(2 * self.scale)
                rect_h = int(width * self.scale)
            else:  # 上方出口
                top_left = self.world_to_screen(np.array([pos[0] - width/2, self.scene_height]))
                rect_w = int(width * self.scale)
                rect_h = int(2 * self.scale)

            pygame.draw.rect(screen, LIGHT_GREEN, (top_left[0], top_left[1], rect_w, rect_h))

            # 出口标签
            label = font.render(exit_info['label'], True, GREEN)
            screen.blit(label, (top_left[0] + 5, top_left[1] + 5))

        # 绘制闸机
        for gate in self.gates:
            pos = gate['position']
            screen_pos = self.world_to_screen(pos)
            gate_w = int(gate['length'] * self.scale)
            gate_h = int(1.5 * self.scale)

            rect = pygame.Rect(
                screen_pos[0] - gate_w//2,
                screen_pos[1] - gate_h//2,
                gate_w, gate_h
            )
            pygame.draw.rect(screen, GATE_COLOR, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)

            # 闸机通道指示
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
            rect_w = int(w * self.scale)
            rect_h = int(h * self.scale)

            rect = pygame.Rect(
                screen_pos[0] - rect_w//2,
                screen_pos[1] - rect_h//2,
                rect_w, rect_h
            )

            if facility['type'] == 'info':
                pygame.draw.rect(screen, (100, 150, 200), rect)
            else:
                pygame.draw.rect(screen, (180, 180, 180), rect)

            pygame.draw.rect(screen, BLACK, rect, 2)

            # 标签
            label = font_small.render(facility['label'], True, BLACK)
            screen.blit(label, (rect.centerx - label.get_width()//2,
                               rect.centery - label.get_height()//2))

        # 绘制墙壁
        wall_color = BLACK
        wall_width = 4

        # 外墙
        walls = [
            # 上墙
            (np.array([0, self.scene_height]), np.array([35, self.scene_height])),
            (np.array([45, self.scene_height]), np.array([self.scene_width, self.scene_height])),
            # 下墙
            (np.array([0, 0]), np.array([self.scene_width, 0])),
            # 左墙
            (np.array([0, 0]), np.array([0, self.scene_height])),
            # 右墙
            (np.array([self.scene_width, 0]), np.array([self.scene_width, 8])),
            (np.array([self.scene_width, 12]), np.array([self.scene_width, 28])),
            (np.array([self.scene_width, 32]), np.array([self.scene_width, self.scene_height])),
        ]

        for start, end in walls:
            pygame.draw.line(screen, wall_color,
                           self.world_to_screen(start),
                           self.world_to_screen(end), wall_width)

        # 绘制行人
        for ped in self.model.pedestrians:
            screen_pos = self.world_to_screen(ped.position)
            radius = int(0.35 * self.scale)

            speed = ped.speed
            if speed < 0.3:
                color = RED
            elif speed < 0.8:
                color = ORANGE
            else:
                color = BLUE

            pygame.draw.circle(screen, color, screen_pos, radius)
            pygame.draw.circle(screen, DARK_BLUE, screen_pos, radius, 1)

        # 绘制区域标签
        platform_label = font.render("站台区", True, (100, 80, 60))
        screen.blit(platform_label, self.world_to_screen(np.array([3, 22])))

        gate_label = font.render("闸机", True, BLACK)
        screen.blit(gate_label, self.world_to_screen(np.array([18, 36])))

        hall_label = font.render("出站大厅", True, (80, 80, 80))
        screen.blit(hall_label, self.world_to_screen(np.array([38, 22])))

        # 信息面板
        self.draw_info_panel(screen, font, font_small)

    def draw_info_panel(self, screen, font, font_small):
        """绘制信息面板"""
        panel_x = int(self.scene_width * self.scale) + 70
        panel_y = 30

        time_elapsed = self.step_count * self.dt * 3
        remaining = len(self.model.pedestrians)

        # 标题
        title = font.render("成都东客站出站口", True, BLACK)
        screen.blit(title, (panel_x, panel_y))

        lines = [
            "",
            f"时间: {time_elapsed:.1f}秒",
            f"已疏散: {self.evacuated_count}",
            f"剩余: {remaining}",
            f"总人数: {self.n_pedestrians}",
            "",
            "各出口疏散:",
            f"  出口A: {self.evacuated_by_exit['A']}",
            f"  出口B: {self.evacuated_by_exit['B']}",
            f"  出口C: {self.evacuated_by_exit['C']}",
            "",
            "== 操作 ==",
            "空格: 暂停/继续",
            "R: 重新开始",
            "ESC: 退出",
            "",
            "== 图例 ==",
        ]

        for i, line in enumerate(lines):
            text = font_small.render(line, True, BLACK)
            screen.blit(text, (panel_x, panel_y + 25 + i * 20))

        # 图例
        legend_y = panel_y + 25 + len(lines) * 20
        legends = [
            (BLUE, "正常行走"),
            (ORANGE, "速度较慢"),
            (RED, "拥堵"),
        ]
        for i, (color, label) in enumerate(legends):
            pygame.draw.circle(screen, color, (panel_x + 10, legend_y + i * 22 + 8), 6)
            text = font_small.render(label, True, BLACK)
            screen.blit(text, (panel_x + 25, legend_y + i * 22))

    def update(self):
        """更新模拟"""
        if self.paused:
            return

        for _ in range(3):
            self.model.step(self.dt)

        # 检查疏散
        to_remove = []
        for ped in self.model.pedestrians:
            for exit_info in self.exits:
                dist = np.linalg.norm(ped.position - exit_info['position'])
                if dist < exit_info['width']:
                    to_remove.append((ped, exit_info['id']))
                    break

        for ped, exit_id in to_remove:
            self.model.pedestrians.remove(ped)
            self.evacuated_count += 1
            self.evacuated_by_exit[exit_id] += 1

        self.step_count += 1

    def run(self):
        """运行可视化"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("成都东客站地铁出站口疏散仿真 - 社会力模型")
        clock = pygame.time.Clock()

        # 尝试加载中文字体
        try:
            font = pygame.font.SysFont('PingFang SC', 18)
            font_small = pygame.font.SysFont('PingFang SC', 14)
        except:
            font = pygame.font.SysFont('Arial', 18)
            font_small = pygame.font.SysFont('Arial', 14)

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
            self.draw_scene(screen, font, font_small)
            pygame.display.flip()
            clock.tick(60)

            if len(self.model.pedestrians) == 0 and not self.paused:
                self.paused = True
                time_elapsed = self.step_count * self.dt * 3
                print(f"\n疏散完成!")
                print(f"总用时: {time_elapsed:.1f}秒")
                print(f"各出口疏散人数: A={self.evacuated_by_exit['A']}, "
                      f"B={self.evacuated_by_exit['B']}, C={self.evacuated_by_exit['C']}")

        pygame.quit()


def main():
    print("=" * 50)
    print("成都东客站地铁出站口疏散仿真")
    print("=" * 50)
    print("\n场景说明:")
    print("  - 站台区: 行人起始位置")
    print("  - 闸机区: 5个闸机通道")
    print("  - 出站大厅: 柱子、信息台、楼梯")
    print("  - 3个出口: A(右下), B(右上), C(上方主出口)")
    print("\n控制:")
    print("  空格 - 暂停/继续")
    print("  R    - 重新开始")
    print("  ESC  - 退出")
    print()

    visualizer = MetroStationVisualizer(
        n_pedestrians=80,
        scale=12.0,
        dt=0.05
    )
    visualizer.run()


if __name__ == "__main__":
    main()
