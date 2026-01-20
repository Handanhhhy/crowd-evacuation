"""
人群疏散可视化动画
使用 matplotlib 动画展示疏散过程
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection

from sfm.social_force import SocialForceModel, Pedestrian, create_random_pedestrians


class EvacuationVisualizer:
    """疏散过程可视化器"""

    def __init__(
        self,
        n_pedestrians: int = 50,
        scene_size: tuple = (30.0, 20.0),
        n_exits: int = 2,
        dt: float = 0.05
    ):
        self.n_pedestrians = n_pedestrians
        self.scene_width, self.scene_height = scene_size
        self.n_exits = n_exits
        self.dt = dt

        # 出口位置
        self.exits = []
        for i in range(n_exits):
            y_pos = (i + 1) * self.scene_height / (n_exits + 1)
            self.exits.append({
                'position': np.array([self.scene_width, y_pos]),
                'width': 2.0
            })

        # 创建模型
        self.model = None
        self.evacuated = []
        self.step_count = 0

        # 动画相关
        self.fig = None
        self.ax = None
        self.ped_circles = []
        self.time_text = None
        self.count_text = None

    def setup_model(self):
        """初始化社会力模型"""
        self.model = SocialForceModel(tau=0.5, A=2000.0, B=0.08)

        # 添加墙壁
        self.model.add_obstacle(
            np.array([0, self.scene_height]),
            np.array([self.scene_width, self.scene_height])
        )
        self.model.add_obstacle(
            np.array([0, 0]),
            np.array([self.scene_width, 0])
        )
        self.model.add_obstacle(
            np.array([0, 0]),
            np.array([0, self.scene_height])
        )

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

        # 添加障碍物（柱子）
        pillar_positions = [
            np.array([15, 7]),
            np.array([15, 13]),
        ]
        for pos in pillar_positions:
            size = 1.0
            self.model.add_obstacle(pos - np.array([size, 0]), pos + np.array([size, 0]))
            self.model.add_obstacle(pos - np.array([0, size]), pos + np.array([0, size]))

        # 添加行人
        for i in range(self.n_pedestrians):
            position = np.array([
                np.random.uniform(2, self.scene_width * 0.4),
                np.random.uniform(2, self.scene_height - 2)
            ])

            # 选择最近的出口作为目标
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

        self.evacuated = []
        self.step_count = 0

    def setup_plot(self):
        """设置绘图"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(-1, self.scene_width + 3)
        self.ax.set_ylim(-1, self.scene_height + 1)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')

        # 绘制墙壁
        wall_color = '#333333'
        wall_width = 3

        # 上下左墙
        self.ax.plot([0, self.scene_width], [0, 0], color=wall_color, linewidth=wall_width)
        self.ax.plot([0, self.scene_width], [self.scene_height, self.scene_height],
                    color=wall_color, linewidth=wall_width)
        self.ax.plot([0, 0], [0, self.scene_height], color=wall_color, linewidth=wall_width)

        # 右墙（带出口间隙）
        prev_y = 0
        for exit_info in sorted(self.exits, key=lambda e: e['position'][1]):
            exit_y = exit_info['position'][1]
            half_w = exit_info['width'] / 2

            if exit_y - half_w > prev_y:
                self.ax.plot([self.scene_width, self.scene_width],
                           [prev_y, exit_y - half_w],
                           color=wall_color, linewidth=wall_width)
            prev_y = exit_y + half_w

        if prev_y < self.scene_height:
            self.ax.plot([self.scene_width, self.scene_width],
                        [prev_y, self.scene_height],
                        color=wall_color, linewidth=wall_width)

        # 标记出口
        for i, exit_info in enumerate(self.exits):
            pos = exit_info['position']
            self.ax.annotate(
                f'Exit {i+1}',
                xy=(pos[0] + 0.5, pos[1]),
                fontsize=10,
                color='green',
                fontweight='bold'
            )
            # 出口区域
            self.ax.fill_between(
                [self.scene_width, self.scene_width + 1],
                pos[1] - exit_info['width']/2,
                pos[1] + exit_info['width']/2,
                color='lightgreen',
                alpha=0.5
            )

        # 绘制障碍物
        for pos in [np.array([15, 7]), np.array([15, 13])]:
            rect = Rectangle(
                (pos[0] - 0.8, pos[1] - 0.8), 1.6, 1.6,
                facecolor='gray',
                edgecolor='black',
                linewidth=2
            )
            self.ax.add_patch(rect)

        # 初始化行人圆点
        self.ped_circles = []
        for ped in self.model.pedestrians:
            circle = Circle(
                ped.position, 0.3,
                facecolor='steelblue',
                edgecolor='darkblue',
                linewidth=0.5,
                alpha=0.8
            )
            self.ax.add_patch(circle)
            self.ped_circles.append(circle)

        # 文本信息
        self.time_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=12, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        self.ax.set_title('Crowd Evacuation Simulation (Social Force Model + RL)',
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        plt.tight_layout()

    def update(self, frame):
        """更新动画帧"""
        # 运行多步物理模拟
        for _ in range(3):
            self.model.step(self.dt)

        # 检查疏散
        to_remove = []
        for ped in self.model.pedestrians:
            for exit_info in self.exits:
                dist = np.linalg.norm(ped.position - exit_info['position'])
                if dist < exit_info['width']:
                    to_remove.append(ped)
                    self.evacuated.append(ped)
                    break

        for ped in to_remove:
            self.model.pedestrians.remove(ped)

        # 更新行人位置
        active_peds = self.model.pedestrians
        for i, circle in enumerate(self.ped_circles):
            if i < len(active_peds):
                ped = active_peds[i]
                circle.center = ped.position

                # 根据速度变色
                speed = ped.speed
                if speed < 0.5:
                    circle.set_facecolor('red')  # 拥堵
                elif speed < 1.0:
                    circle.set_facecolor('orange')  # 缓慢
                else:
                    circle.set_facecolor('steelblue')  # 正常

                circle.set_visible(True)
            else:
                circle.set_visible(False)

        self.step_count += 1
        time_elapsed = self.step_count * self.dt * 3

        # 更新文本
        remaining = len(self.model.pedestrians)
        evacuated = len(self.evacuated)
        self.time_text.set_text(
            f'Time: {time_elapsed:.1f}s\n'
            f'Evacuated: {evacuated}/{self.n_pedestrians}\n'
            f'Remaining: {remaining}'
        )

        return self.ped_circles + [self.time_text]

    def run(self, save_path: str = None):
        """运行动画"""
        self.setup_model()
        self.setup_plot()

        # 创建动画
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=500,
            interval=50,
            blit=True,
            repeat=False
        )

        if save_path:
            print(f"保存动画到: {save_path}")
            ani.save(save_path, writer='pillow', fps=20)
            print("保存完成!")

        plt.show()


def main():
    print("=" * 50)
    print("人群疏散可视化演示")
    print("=" * 50)

    visualizer = EvacuationVisualizer(
        n_pedestrians=60,
        scene_size=(30.0, 20.0),
        n_exits=2,
        dt=0.05
    )

    # 保存为GIF
    save_path = str(project_root / "outputs" / "figures" / "evacuation_animation.gif")
    visualizer.run(save_path=save_path)


if __name__ == "__main__":
    main()
