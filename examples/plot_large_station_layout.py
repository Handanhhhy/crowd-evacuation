"""
成都东站大型地铁站场景布局图 (matplotlib版本)
T形布局 - 按原图绘制

结构说明:
- 左上角: 步梯 + 扶梯 (并列)
- 左下角: 步梯 + 扶梯 (并列) + 直升电梯(右侧)
- 左侧: 闸机a (疏散口)
- 右侧: 闸机子 (疏散口)
- 上排: 闸机b/c/d (疏散口)
- 下排: 闸机e/f/g (疏散口)
- 中间: 3个扶梯 (涌入点)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import platform

# 设置中文字体
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_t_shape_station(save_path="outputs/figures/large_station_layout.png"):
    """绘制T形场景布局图 - 按原图"""

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # 设置坐标轴
    ax.set_xlim(-15, 165)
    ax.set_ylim(-10, 90)
    ax.set_aspect('equal')
    ax.set_xlabel('X (米)', fontsize=12)
    ax.set_ylabel('Y (米)', fontsize=12)
    ax.set_title('成都东站地铁站疏散场景布局 (T形结构)', fontsize=16, fontweight='bold')

    # 网格
    ax.grid(True, linestyle='--', alpha=0.2)

    # ==================== T形外墙轮廓 ====================
    # T形外轮廓 (只画外墙，内部完全连通)

    # 左侧走廊外墙
    ax.plot([0, 0], [0, 80], 'k-', linewidth=3)  # 左墙
    ax.plot([0, 20], [80, 80], 'k-', linewidth=3)  # 左上角顶
    ax.plot([0, 20], [0, 0], 'k-', linewidth=3)  # 左下角底

    # 上部走廊外墙
    ax.plot([20, 20], [80, 70], 'k-', linewidth=3)  # 左上角内侧垂直
    ax.plot([20, 150], [70, 70], 'k-', linewidth=3)  # 上边
    ax.plot([150, 150], [70, 10], 'k-', linewidth=3)  # 右侧完整垂直墙

    # 下部走廊外墙
    ax.plot([20, 20], [0, 10], 'k-', linewidth=3)  # 左下角内侧垂直
    ax.plot([20, 150], [10, 10], 'k-', linewidth=3)  # 下边

    # 填充T形区域背景 (浅灰色) - 整个内部连通
    from matplotlib.patches import Polygon
    t_shape = Polygon([
        (0, 0), (20, 0), (20, 10), (150, 10), (150, 70), (20, 70), (20, 80), (0, 80)
    ], facecolor='#f8f8f8', edgecolor='none')
    ax.add_patch(t_shape)

    # ==================== 左上角: 步梯 + 扶梯 ====================
    # 扶梯/步梯是障碍物（斜线填充），行人从下方边缘生成

    # 步梯 - 中心(5, 71)，尺寸6x12
    stair_upper = FancyBboxPatch((2, 65), 6, 12, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='navy', linewidth=2,
                                 hatch='///')  # 斜线表示障碍物
    ax.add_patch(stair_upper)
    ax.text(5, 71, '步梯', ha='center', va='center', fontsize=9, color='navy', fontweight='bold')
    # 下方生成边缘（绿色线）
    ax.plot([2, 8], [65, 65], color='limegreen', linewidth=4)
    # 行人流向箭头
    ax.annotate('', xy=(5, 62), xytext=(5, 65),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 扶梯 - 中心(11, 71)，紧挨步梯
    escalator_upper = FancyBboxPatch((8, 65), 6, 12, boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor='gray', linewidth=2,
                                     hatch='///')
    ax.add_patch(escalator_upper)
    ax.text(11, 71, '扶梯', ha='center', va='center', fontsize=9, color='dimgray', fontweight='bold')
    # 下方生成边缘
    ax.plot([8, 14], [65, 65], color='limegreen', linewidth=4)
    ax.annotate('', xy=(11, 62), xytext=(11, 65),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # ==================== 左下角: 步梯 + 扶梯 + 直升电梯 ====================
    # 步梯和扶梯是障碍物，行人从上方边缘出来

    # 步梯 - 中心(5, 9)，尺寸6x12
    stair_lower = FancyBboxPatch((2, 3), 6, 12, boxstyle="round,pad=0.02",
                                 facecolor='white', edgecolor='navy', linewidth=2,
                                 hatch='///')
    ax.add_patch(stair_lower)
    ax.text(5, 9, '步梯', ha='center', va='center', fontsize=9, color='navy', fontweight='bold')
    # 上方出口边缘（绿色线）
    ax.plot([2, 8], [15, 15], color='limegreen', linewidth=4)
    # 行人流向箭头
    ax.annotate('', xy=(5, 18), xytext=(5, 15),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 扶梯 - 中心(11, 9)，紧挨步梯
    escalator_lower = FancyBboxPatch((8, 3), 6, 12, boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor='gray', linewidth=2,
                                     hatch='///')
    ax.add_patch(escalator_lower)
    ax.text(11, 9, '扶梯', ha='center', va='center', fontsize=9, color='dimgray', fontweight='bold')
    # 上方出口边缘
    ax.plot([8, 14], [15, 15], color='limegreen', linewidth=4)
    ax.annotate('', xy=(11, 18), xytext=(11, 15),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 直升电梯 (红色斜线 - 纯障碍物，无出口)
    elevator = FancyBboxPatch((20, 26), 8, 8, boxstyle="round,pad=0.02",
                              facecolor='white', edgecolor='darkred', linewidth=2,
                              hatch='xxx')
    ax.add_patch(elevator)
    ax.text(24, 30, '直升\n电梯', ha='center', va='center', fontsize=8, color='darkred', fontweight='bold')

    # ==================== 闸机a (左侧疏散口) - Y=40为中心对称 ====================

    gate_a = Rectangle((0, 30), 3, 20, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_a)
    ax.text(-6, 40, '闸机a\n(疏散口)', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    # 通道线
    for i in range(1, 5):
        y = 30 + i * 4
        ax.plot([0, 3], [y, y], 'darkgreen', linewidth=0.5)
    # 箭头表示疏散方向
    for i in range(5):
        y = 32 + i * 4
        ax.annotate('', xy=(-2, y), xytext=(2, y),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # ==================== 闸机子 (右侧疏散口) - Y=40为中心对称 ====================

    gate_zi = Rectangle((147, 30), 3, 20, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_zi)
    ax.text(156, 40, '闸机子\n(疏散口)', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    # 通道线
    for i in range(1, 5):
        y = 30 + i * 4
        ax.plot([147, 150], [y, y], 'darkgreen', linewidth=0.5)
    # 箭头表示疏散方向
    for i in range(5):
        y = 32 + i * 4
        ax.annotate('', xy=(152, y), xytext=(148, y),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # ==================== 上排闸机 (疏散口) ====================

    # 闸机b
    gate_b_x = 45
    gate_b = Rectangle((gate_b_x, 67), 15, 3, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_b)
    ax.text(gate_b_x + 7.5, 73, '闸机b', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    for i in range(1, 5):
        x = gate_b_x + i * 3
        ax.plot([x, x], [67, 70], 'darkgreen', linewidth=0.5)
    # 箭头
    for i in range(5):
        x = gate_b_x + 1.5 + i * 3
        ax.annotate('', xy=(x, 72), xytext=(x, 68),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))

    # 闸机c
    gate_c_x = 75
    gate_c = Rectangle((gate_c_x, 67), 15, 3, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_c)
    ax.text(gate_c_x + 7.5, 73, '闸机c', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    for i in range(1, 5):
        x = gate_c_x + i * 3
        ax.plot([x, x], [67, 70], 'darkgreen', linewidth=0.5)
    for i in range(5):
        x = gate_c_x + 1.5 + i * 3
        ax.annotate('', xy=(x, 72), xytext=(x, 68),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))

    # 闸机d
    gate_d_x = 105
    gate_d = Rectangle((gate_d_x, 67), 15, 3, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_d)
    ax.text(gate_d_x + 7.5, 73, '闸机d', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    for i in range(1, 5):
        x = gate_d_x + i * 3
        ax.plot([x, x], [67, 70], 'darkgreen', linewidth=0.5)
    for i in range(5):
        x = gate_d_x + 1.5 + i * 3
        ax.annotate('', xy=(x, 72), xytext=(x, 68),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))

    # ==================== 下排闸机 (疏散口) ====================

    # 闸机e
    gate_e_x = 45
    gate_e = Rectangle((gate_e_x, 10), 15, 3, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_e)
    ax.text(gate_e_x + 7.5, 6, '闸机e', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    for i in range(1, 5):
        x = gate_e_x + i * 3
        ax.plot([x, x], [10, 13], 'darkgreen', linewidth=0.5)
    for i in range(5):
        x = gate_e_x + 1.5 + i * 3
        ax.annotate('', xy=(x, 8), xytext=(x, 12),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))

    # 闸机f
    gate_f_x = 75
    gate_f = Rectangle((gate_f_x, 10), 15, 3, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_f)
    ax.text(gate_f_x + 7.5, 6, '闸机f', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    for i in range(1, 5):
        x = gate_f_x + i * 3
        ax.plot([x, x], [10, 13], 'darkgreen', linewidth=0.5)
    for i in range(5):
        x = gate_f_x + 1.5 + i * 3
        ax.annotate('', xy=(x, 8), xytext=(x, 12),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))

    # 闸机g
    gate_g_x = 105
    gate_g = Rectangle((gate_g_x, 10), 15, 3, facecolor='limegreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(gate_g)
    ax.text(gate_g_x + 7.5, 6, '闸机g', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
    for i in range(1, 5):
        x = gate_g_x + i * 3
        ax.plot([x, x], [10, 13], 'darkgreen', linewidth=0.5)
    for i in range(5):
        x = gate_g_x + 1.5 + i * 3
        ax.annotate('', xy=(x, 8), xytext=(x, 12),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))

    # ==================== 中间扶梯 (涌入点) ====================
    # 扶梯是障碍物（斜线填充），横向布局，行人从左侧出口出来
    # 尺寸: 12m × 6m (横向)

    esc_w, esc_h = 12, 6  # 横向扶梯

    # 扶梯1 - 中心(45, 40)
    esc1_x = 45 - esc_w/2
    esc1_y = 40 - esc_h/2
    esc1 = FancyBboxPatch((esc1_x, esc1_y), esc_w, esc_h, boxstyle="round,pad=0.02",
                          facecolor='white', edgecolor='orange', linewidth=2,
                          hatch='///')
    ax.add_patch(esc1)
    ax.text(45, 40, '扶梯1', ha='center', va='center', fontsize=9, color='darkorange', fontweight='bold')
    # 左侧出口边缘（绿色线）
    ax.plot([esc1_x, esc1_x], [esc1_y, esc1_y + esc_h], color='limegreen', linewidth=4)
    # 行人流向箭头
    ax.annotate('', xy=(esc1_x - 5, 40), xytext=(esc1_x, 40),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 扶梯2 - 中心(75, 40)
    esc2_x = 75 - esc_w/2
    esc2_y = 40 - esc_h/2
    esc2 = FancyBboxPatch((esc2_x, esc2_y), esc_w, esc_h, boxstyle="round,pad=0.02",
                          facecolor='white', edgecolor='orange', linewidth=2,
                          hatch='///')
    ax.add_patch(esc2)
    ax.text(75, 40, '扶梯2', ha='center', va='center', fontsize=9, color='darkorange', fontweight='bold')
    ax.plot([esc2_x, esc2_x], [esc2_y, esc2_y + esc_h], color='limegreen', linewidth=4)
    ax.annotate('', xy=(esc2_x - 5, 40), xytext=(esc2_x, 40),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 扶梯3 - 中心(105, 40)
    esc3_x = 105 - esc_w/2
    esc3_y = 40 - esc_h/2
    esc3 = FancyBboxPatch((esc3_x, esc3_y), esc_w, esc_h, boxstyle="round,pad=0.02",
                          facecolor='white', edgecolor='orange', linewidth=2,
                          hatch='///')
    ax.add_patch(esc3)
    ax.text(105, 40, '扶梯3', ha='center', va='center', fontsize=9, color='darkorange', fontweight='bold')
    ax.plot([esc3_x, esc3_x], [esc3_y, esc3_y + esc_h], color='limegreen', linewidth=4)
    ax.annotate('', xy=(esc3_x - 5, 40), xytext=(esc3_x, 40),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # ==================== 人流说明 ====================

    # 涌入说明
    ax.annotate('下层人员\n持续涌入', xy=(82.5, 40), xytext=(82.5, -5),
               fontsize=11, ha='center', color='darkorange', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))

    # ==================== 图例 ====================

    legend_x = 155
    legend_y = 80

    ax.text(legend_x, legend_y, '图例', fontsize=12, fontweight='bold')

    legend_items = [
        (legend_y - 8, 'limegreen', '闸机 (疏散口)'),
        (legend_y - 16, 'limegreen', '行人生成边缘'),
    ]

    for y, color, label in legend_items:
        if color == 'limegreen' and '生成' in label:
            ax.plot([legend_x, legend_x + 4], [y + 1.5, y + 1.5], color=color, linewidth=4)
        else:
            rect = Rectangle((legend_x, y), 4, 3)
            rect.set_facecolor(color)
            rect.set_edgecolor('gray')
            ax.add_patch(rect)
        ax.text(legend_x + 6, y + 1.5, label, fontsize=9, va='center')

    # 斜线图例
    hatch_items = [
        (legend_y - 24, '///', 'navy', '步梯/扶梯 (障碍物)'),
        (legend_y - 32, 'xxx', 'darkred', '电梯 (禁用)'),
    ]
    for y, hatch, edgecolor, label in hatch_items:
        rect = FancyBboxPatch((legend_x, y), 4, 3, boxstyle="round,pad=0.01",
                              facecolor='white', edgecolor=edgecolor, hatch=hatch)
        ax.add_patch(rect)
        ax.text(legend_x + 6, y + 1.5, label, fontsize=9, va='center')

    # 删除原来的legend循环，用新的替代
    legend_items = []  # 清空避免重复

    for y, color, label in legend_items:
        rect = Rectangle((legend_x, y), 4, 3)
        rect.set_facecolor(color)
        rect.set_edgecolor('gray')
        ax.add_patch(rect)
        ax.text(legend_x + 6, y + 1.5, label, fontsize=9, va='center')

    # 信息框
    info_text = """队列模型:
• 扶梯/步梯: 障碍物
• 绿色线: 出口边缘
• 行人排队→运送→出口

涌入点 (7个):
  中间扶梯×3: 左侧出口
  左上步梯+扶梯: 下方出口
  左下步梯+扶梯: 上方出口

人流量:
  小: 500+500=1000人
  中: 1000+1000=2000人
  大: 1500+1500=3000人"""

    ax.text(legend_x, legend_y - 50, info_text, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # 保存图片
    from pathlib import Path
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"T形场景布局图已保存: {save_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    plot_t_shape_station()
