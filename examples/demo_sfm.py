"""
社会力模型演示脚本
运行此脚本可视化人群疏散过程
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

from sfm.social_force import SocialForceModel, Pedestrian, create_random_pedestrians


def run_simulation():
    """运行人群疏散仿真"""

    # 创建社会力模型
    model = SocialForceModel(
        tau=0.5,
        A=2000.0,
        B=0.08
    )

    # 场景参数
    scene_width = 30.0
    scene_height = 20.0
    exit_position = np.array([scene_width, scene_height / 2])

    # 添加行人
    pedestrians = create_random_pedestrians(
        n=50,
        spawn_area=(2, 2, 15, 18),
        target=exit_position,
        seed=42
    )

    for ped in pedestrians:
        model.add_pedestrian(ped)

    # 添加墙壁
    # 上墙
    model.add_obstacle(np.array([0, scene_height]), np.array([scene_width, scene_height]))
    # 下墙
    model.add_obstacle(np.array([0, 0]), np.array([scene_width, 0]))
    # 左墙
    model.add_obstacle(np.array([0, 0]), np.array([0, scene_height]))
    # 右墙（带出口）- 分成两段
    exit_half_width = 2.0
    model.add_obstacle(
        np.array([scene_width, 0]),
        np.array([scene_width, scene_height/2 - exit_half_width])
    )
    model.add_obstacle(
        np.array([scene_width, scene_height/2 + exit_half_width]),
        np.array([scene_width, scene_height])
    )

    # 添加一个障碍物（柱子）
    pillar_pos = np.array([20, 10])
    pillar_size = 1.5
    model.add_obstacle(
        pillar_pos - np.array([pillar_size/2, 0]),
        pillar_pos + np.array([pillar_size/2, 0])
    )
    model.add_obstacle(
        pillar_pos - np.array([0, pillar_size/2]),
        pillar_pos + np.array([0, pillar_size/2])
    )

    # 仿真参数
    dt = 0.05
    max_steps = 2000

    # 记录轨迹
    trajectories = {ped.id: [ped.position.copy()] for ped in pedestrians}
    arrived_count = []

    print("开始仿真...")

    for step in range(max_steps):
        model.step(dt)

        # 记录轨迹
        for ped in model.pedestrians:
            trajectories[ped.id].append(ped.position.copy())

        # 统计到达出口的人数
        arrived = sum(
            1 for ped in model.pedestrians
            if np.linalg.norm(ped.target - ped.position) < 1.5
        )
        arrived_count.append(arrived)

        if step % 100 == 0:
            print(f"Step {step}, 到达出口: {arrived}/{len(pedestrians)}")

        if model.is_finished(threshold=1.5):
            print(f"所有行人已到达出口! 用时: {step * dt:.1f} 秒")
            break

    return model, trajectories, arrived_count, (scene_width, scene_height), exit_position


def visualize_result(model, trajectories, arrived_count, scene_size, exit_pos):
    """可视化结果"""

    scene_width, scene_height = scene_size

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：轨迹
    ax1 = axes[0]
    ax1.set_xlim(-1, scene_width + 2)
    ax1.set_ylim(-1, scene_height + 1)
    ax1.set_aspect('equal')
    ax1.set_title('行人轨迹')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')

    # 绘制墙壁
    ax1.plot([0, scene_width], [0, 0], 'k-', linewidth=2)
    ax1.plot([0, scene_width], [scene_height, scene_height], 'k-', linewidth=2)
    ax1.plot([0, 0], [0, scene_height], 'k-', linewidth=2)

    # 出口
    exit_half = 2.0
    ax1.plot([scene_width, scene_width], [0, scene_height/2 - exit_half], 'k-', linewidth=2)
    ax1.plot([scene_width, scene_width], [scene_height/2 + exit_half, scene_height], 'k-', linewidth=2)

    # 标记出口
    ax1.annotate('出口', xy=(scene_width + 0.5, scene_height/2), fontsize=10, color='green')

    # 绘制轨迹
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    for (ped_id, traj), color in zip(trajectories.items(), colors):
        traj = np.array(traj)
        ax1.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.5, linewidth=0.5)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=3)  # 起点

    # 右图：到达人数曲线
    ax2 = axes[1]
    ax2.plot(arrived_count, 'b-', linewidth=1.5)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('到达出口人数')
    ax2.set_title('疏散进度')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=len(trajectories), color='r', linestyle='--', label=f'总人数 ({len(trajectories)})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent.parent / 'outputs' / 'figures' / 'sfm_demo.png'), dpi=150)
    print("图像已保存到 outputs/figures/sfm_demo.png")
    plt.show()


def main():
    print("=" * 50)
    print("社会力模型人群疏散演示")
    print("=" * 50)

    model, trajectories, arrived_count, scene_size, exit_pos = run_simulation()
    visualize_result(model, trajectories, arrived_count, scene_size, exit_pos)


if __name__ == "__main__":
    main()
