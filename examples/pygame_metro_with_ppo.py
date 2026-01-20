"""
成都东客站地铁出站口疏散仿真 - 集成PPO智能引导
社会力模型 + PPO强化学习动态优化出口选择
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pygame
from stable_baselines3 import PPO
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
PURPLE = (147, 112, 219)
BG_COLOR = (240, 240, 235)
FLOOR_COLOR = (220, 220, 210)
GATE_COLOR = (100, 100, 100)


class MetroStationWithPPO:
    """成都东客站地铁出站口 - PPO智能引导版"""

    def __init__(
        self,
        n_pedestrians: int = 80,
        scale: float = 12.0,
        dt: float = 0.05,
        use_ppo: bool = True
    ):
        self.n_pedestrians = n_pedestrians
        self.scale = scale
        self.dt = dt
        self.use_ppo = use_ppo

        # 场景尺寸
        self.scene_width = 60.0
        self.scene_height = 40.0

        # 窗口大小
        self.window_width = int(self.scene_width * scale) + 220
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

        # 出口定义
        self.exits = [
            {'id': 0, 'name': 'A', 'position': np.array([60, 10]), 'width': 4.0, 'label': '出口A'},
            {'id': 1, 'name': 'B', 'position': np.array([60, 30]), 'width': 4.0, 'label': '出口B'},
            {'id': 2, 'name': 'C', 'position': np.array([40, 40]), 'width': 5.0, 'label': '出口C (主出口)'},
        ]

        # 柱子位置
        self.pillars = [
            np.array([30, 12]), np.array([30, 28]),
            np.array([45, 12]), np.array([45, 28]),
            np.array([35, 20]), np.array([50, 20]),
        ]

        # 设施
        self.facilities = [
            {'type': 'info', 'position': np.array([32, 20]), 'size': (3, 2), 'label': '信息台'},
            {'type': 'stairs', 'position': np.array([55, 20]), 'size': (4, 8), 'label': '楼梯'},
        ]

        # PPO模型
        self.ppo_model = None
        self.current_action = 0  # 当前PPO推荐的出口
        self.ppo_update_interval = 10  # 每10步更新一次PPO决策

        # 状态
        self.model = None
        self.evacuated_count = 0
        self.evacuated_by_exit = {'A': 0, 'B': 0, 'C': 0}
        self.step_count = 0
        self.running = True
        self.paused = False

        # 对比模式
        self.show_comparison = False

    def load_ppo_model(self):
        """加载训练好的PPO模型"""
        model_path = project_root / "outputs" / "models" / "ppo_evacuation.zip"

        if model_path.exists() and self.use_ppo:
            try:
                self.ppo_model = PPO.load(str(model_path))
                print(f"PPO模型已加载: {model_path}")
                return True
            except Exception as e:
                print(f"加载PPO模型失败: {e}")
                self.ppo_model = None
                return False
        else:
            print("未找到PPO模型或未启用PPO")
            return False

    def world_to_screen(self, pos):
        """世界坐标转屏幕坐标"""
        x = int(pos[0] * self.scale) + 50
        y = int((self.scene_height - pos[1]) * self.scale) + 30
        return (x, y)

    def setup_model(self):
        """初始化社会力模型"""
        self.model = SocialForceModel(tau=0.5, A=2000.0, B=0.08)

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

        # 添加行人
        np.random.seed(None)
        for i in range(self.n_pedestrians):
            position = np.array([
                np.random.uniform(2, 14),
                np.random.uniform(12, 28)
            ])
            target = self._choose_initial_exit(position)

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
        """获取PPO模型的观测状态"""
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

        # 构建观测 (与训练时的环境保持一致)
        # 注意：训练环境是2个出口，这里是3个出口，需要适配
        # 简化处理：取前2个出口的数据
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
        """应用PPO引导策略"""
        if self.ppo_model is None:
            return

        # 获取观测
        obs = self.get_ppo_observation()

        # PPO决策
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        self.current_action = int(action)

        # 将PPO的出口选择映射到3个出口
        # action=0 -> 出口A, action=1 -> 出口B
        # 额外逻辑：如果出口A/B拥堵，引导去出口C
        recommended_exit = self.current_action

        # 检查推荐出口的拥堵情况
        _, congestion = self._compute_exit_metrics(self.exits[recommended_exit]['position'])
        if congestion > 0.6:
            # 拥堵时考虑出口C
            recommended_exit = 2

        # 引导部分行人改变目标
        target_pos = self.exits[recommended_exit]['position']

        for ped in self.model.pedestrians:
            # 只有在大厅区域的行人才响应引导
            if ped.position[0] > 22:
                # 根据距离和随机因素决定是否听从引导
                dist_to_current = np.linalg.norm(ped.position - ped.target)
                dist_to_recommended = np.linalg.norm(ped.position - target_pos)

                # 如果推荐出口更近或当前出口拥堵，有更高概率改变
                if dist_to_recommended < dist_to_current * 1.2:
                    prob = 0.15  # 15%的概率响应引导
                else:
                    prob = 0.05

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
            speed = ped.speed

            if speed < 0.3:
                color = RED
            elif speed < 0.8:
                color = ORANGE
            else:
                color = BLUE

            pygame.draw.circle(screen, color, screen_pos, radius)
            pygame.draw.circle(screen, DARK_BLUE, screen_pos, radius, 1)

        # 区域标签
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
        panel_y = 20

        time_elapsed = self.step_count * self.dt * 3
        remaining = len(self.model.pedestrians)

        # 标题
        title = font.render("成都东客站出站口", True, BLACK)
        screen.blit(title, (panel_x, panel_y))

        # PPO状态
        if self.ppo_model is not None and self.use_ppo:
            ppo_status = font_small.render("PPO智能引导: ON", True, PURPLE)
            recommended = self.exits[min(self.current_action, 2)]['name']
            ppo_action = font_small.render(f"推荐出口: {recommended}", True, PURPLE)
        else:
            ppo_status = font_small.render("PPO智能引导: OFF", True, GRAY)
            ppo_action = font_small.render("(按P键开启)", True, GRAY)

        screen.blit(ppo_status, (panel_x, panel_y + 25))
        screen.blit(ppo_action, (panel_x, panel_y + 45))

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
            "P: 开关PPO引导",
            "R: 重新开始",
            "ESC: 退出",
            "",
            "== 图例 ==",
        ]

        for i, line in enumerate(lines):
            text = font_small.render(line, True, BLACK)
            screen.blit(text, (panel_x, panel_y + 65 + i * 18))

        # 图例
        legend_y = panel_y + 65 + len(lines) * 18
        legends = [
            (BLUE, "正常行走"),
            (ORANGE, "速度较慢"),
            (RED, "拥堵"),
            (PURPLE, "PPO推荐出口"),
        ]
        for i, (color, label) in enumerate(legends):
            pygame.draw.circle(screen, color, (panel_x + 10, legend_y + i * 20 + 8), 6)
            text = font_small.render(label, True, BLACK)
            screen.blit(text, (panel_x + 25, legend_y + i * 20))

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

        self.step_count += 1

    def run(self):
        """运行可视化"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("成都东客站疏散仿真 - 社会力模型 + PPO智能引导")
        clock = pygame.time.Clock()

        try:
            font = pygame.font.SysFont('PingFang SC', 18)
            font_small = pygame.font.SysFont('PingFang SC', 14)
        except:
            font = pygame.font.SysFont('Arial', 18)
            font_small = pygame.font.SysFont('Arial', 14)

        # 加载PPO模型
        self.load_ppo_model()
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
                        # 切换PPO开关
                        if self.ppo_model is not None:
                            self.use_ppo = not self.use_ppo
                            print(f"PPO引导: {'开启' if self.use_ppo else '关闭'}")

            self.update()
            self.draw_scene(screen, font, font_small)
            pygame.display.flip()
            clock.tick(60)

            if len(self.model.pedestrians) == 0 and not self.paused:
                self.paused = True
                time_elapsed = self.step_count * self.dt * 3
                print(f"\n疏散完成!")
                print(f"总用时: {time_elapsed:.1f}秒")
                print(f"各出口: A={self.evacuated_by_exit['A']}, "
                      f"B={self.evacuated_by_exit['B']}, C={self.evacuated_by_exit['C']}")
                print(f"PPO引导: {'开启' if self.use_ppo else '关闭'}")

        pygame.quit()


def main():
    print("=" * 50)
    print("成都东客站疏散仿真 - PPO智能引导版")
    print("=" * 50)
    print("\n模型说明:")
    print("  - 社会力模型(SFM): 行人运动仿真")
    print("  - PPO强化学习: 动态引导出口选择")
    print("\n控制:")
    print("  空格 - 暂停/继续")
    print("  P    - 开关PPO智能引导")
    print("  R    - 重新开始")
    print("  ESC  - 退出")
    print()

    visualizer = MetroStationWithPPO(
        n_pedestrians=80,
        scale=12.0,
        dt=0.05,
        use_ppo=True
    )
    visualizer.run()


if __name__ == "__main__":
    main()
