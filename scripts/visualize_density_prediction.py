#!/usr/bin/env python
"""
密度场预测可视化工具

使用 Pygame 可视化密度场预测结果，包括：
- 当前密度场（输入）
- 预测密度场（模型输出）
- 真实密度场（目标）
- 误差热力图

使用方法:
    python scripts/visualize_density_prediction.py --model-path outputs/models/density_predictor.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import pygame
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from prediction import (
    DensityFieldPredictor,
    DensityDataCollector,
    DensityPredictorNet,
    DensityPredictorLite,
    GRID_SIZE,
    MAX_SAFE_DENSITY,
)
from prediction.data_collector import create_dataloader


# 颜色定义
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (128, 128, 128)
BG_COLOR = (245, 245, 245)
TEXT_COLOR = (50, 50, 50)

# 密度颜色映射（改进版 - 热力图风格）
def density_to_color(density, max_val=1.0):
    """将密度值映射到热力图颜色"""
    # 归一化
    val = np.clip(density / max_val, 0.0, 1.0)
    
    # 颜色点 (值, (R, G, B)) - 类似于 Jet/Turbo 
    colors = [
        (0.0, (0, 0, 128)),    # 深蓝 (底色)
        (0.2, (0, 0, 255)),    # 蓝
        (0.4, (0, 255, 255)),  # 青
        (0.6, (0, 255, 0)),    # 绿
        (0.8, (255, 255, 0)),  # 黄
        (1.0, (255, 0, 0))     # 红
    ]
    
    # 找到区间
    for i in range(len(colors) - 1):
        if val <= colors[i+1][0]:
            c1 = colors[i]
            c2 = colors[i+1]
            ratio = (val - c1[0]) / (c2[0] - c1[0])
            
            r = int(c1[1][0] + ratio * (c2[1][0] - c1[1][0]))
            g = int(c1[1][1] + ratio * (c2[1][1] - c1[1][1]))
            b = int(c1[1][2] + ratio * (c2[1][2] - c1[1][2]))
            return (r, g, b)
            
    return colors[-1][1]


class DensityFieldVisualizer:
    """密度场预测可视化器"""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "outputs/training_data",
        use_lite_model: bool = False,
        cell_size: int = 20,  # 每个网格的像素大小
        margin: int = 50,  # 边距
    ):
        """
        Args:
            model_path: 模型路径
            data_dir: 数据目录
            cell_size: 每个网格的像素大小
            margin: 边距
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.use_lite_model = use_lite_model
        self.cell_size = cell_size
        self.margin = margin
        
        # 网格尺寸
        self.grid_w, self.grid_h = GRID_SIZE
        
        # 窗口尺寸（3个并排显示：当前、预测、真实）
        panel_width = self.grid_w * cell_size
        panel_height = self.grid_h * cell_size
        self.panel_width = panel_width
        self.panel_height = panel_height
        
        self.window_width = panel_width * 3 + margin * 4
        self.window_height = panel_height + margin * 3 + 100  # 额外空间显示信息
        
        # 加载模型和数据
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.test_loader = self._load_data()
        
        # 当前状态
        self.current_batch_idx = 0
        self.current_sample_idx = 0
        self.samples = []
        self.paused = False
        self.running = True
        self.show_error = False  # 是否显示误差图
        self.auto_scale = True   # 是否自动缩放颜色范围
        
        # 加载所有样本到内存
        self._load_samples()
        
    def _load_model(self):
        """加载预测模型"""
        print(f"[Visualizer] 加载模型: {self.model_path}")
        
        if self.use_lite_model:
            model = DensityPredictorLite(
                input_channels=4,
                hidden_channels=32,
                grid_size=GRID_SIZE,
            )
        else:
            model = DensityPredictorNet(
                input_channels=4,
                hidden_channels=64,
                grid_size=GRID_SIZE,
            )
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        model = model.to(self.device)
        model.eval()
        
        print(f"[Visualizer] 模型已加载到: {self.device}")
        return model
    
    def _load_data(self):
        """加载测试数据"""
        print(f"[Visualizer] 加载数据: {self.data_dir}")
        
        exits = [{'id': f'exit_{i}', 'position': np.array([0, 0])} for i in range(8)]
        collector = DensityDataCollector(exits=exits, save_dir=self.data_dir)
        collector.load_all_episodes()
        
        _, test_dataset = collector.build_dataset(
            seq_length=10,
            pred_horizon=50,
            train_ratio=0.0,  # 全部用于测试
        )
        
        test_loader = create_dataloader(test_dataset, batch_size=1, shuffle=False)
        
        print(f"[Visualizer] 测试样本数: {len(test_dataset)}")
        return test_loader
    
    def _load_samples(self):
        """加载所有样本到内存"""
        print("[Visualizer] 加载样本...")
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.test_loader, desc="加载样本"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 预测
                pred, _ = self.model(batch_x)
                
                # 转换为numpy
                input_seq = batch_x[0].cpu().numpy()  # [seq_len, 4, h, w]
                current_density = input_seq[-1, 0, :, :]  # 最后一帧的密度
                pred_density = pred[0, 0, :, :].cpu().numpy()
                true_density = batch_y[0, 0, :, :].cpu().numpy()
                
                # 计算误差
                error = np.abs(pred_density - true_density)
                
                self.samples.append({
                    'current': current_density,
                    'predicted': pred_density,
                    'true': true_density,
                    'error': error,
                })
        
        print(f"[Visualizer] 已加载 {len(self.samples)} 个样本")
    
    def _draw_density_field(
        self,
        screen: pygame.Surface,
        density: np.ndarray,
        x_offset: int,
        y_offset: int,
        title: str,
        font: pygame.font.Font = None,
        max_val: float = 1.0,
    ):
        """绘制密度场"""
        if font is None:
            font = pygame.font.Font(None, 16)
        
        # 绘制标题
        title_surface = font.render(f"{title} (Max: {np.max(density):.2f})", True, TEXT_COLOR)
        screen.blit(title_surface, (x_offset, y_offset - 25))
        
        # 绘制网格
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                x = x_offset + i * self.cell_size
                y = y_offset + j * self.cell_size
                
                # 获取密度值
                d = density[i, j]
                color = density_to_color(d, max_val)
                
                # 绘制单元格
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRAY, rect, 1)
                
                # 显示密度值（可选，如果单元格足够大）
                if self.cell_size >= 15:
                    density_text = f"{d:.2f}"
                    text_surface = font.render(density_text, True, WHITE if d > 0.5 else BLACK)
                    text_rect = text_surface.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                    screen.blit(text_surface, text_rect)
    
    def _draw_error_field(
        self,
        screen: pygame.Surface,
        error: np.ndarray,
        x_offset: int,
        y_offset: int,
        title: str,
        font: pygame.font.Font = None,
    ):
        """绘制误差场"""
        if font is None:
            font = pygame.font.Font(None, 16)
        
        # 绘制标题
        title_surface = font.render(title, True, TEXT_COLOR)
        screen.blit(title_surface, (x_offset + self.panel_width // 2 - title_surface.get_width() // 2, y_offset - 25))
        
        # 归一化误差到[0, 1]用于显示
        max_error = np.max(error) if np.max(error) > 0 else 1.0
        error_norm = error / max_error
        
        # 绘制网格
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                x = x_offset + i * self.cell_size
                y = y_offset + j * self.cell_size
                
                # 误差越大，颜色越红
                e = error_norm[i, j]
                color = (int(255 * e), int(255 * (1 - e)), 0)  # 绿色到红色
                
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRAY, rect, 1)
    
    def _draw_info_panel(self, screen: pygame.Surface, font: pygame.font.Font):
        """绘制信息面板"""
        y = self.panel_height + self.margin * 2
        
        info_lines = [
            f"Sample: {self.current_sample_idx + 1} / {len(self.samples)}",
            f"Status: {'Paused' if self.paused else 'Playing'}",
            f"Mode: {'Error Map' if self.show_error else 'Density Fields'}",
            "",
            "Controls:",
            "  [Space] Pause/Play",
            "  [Right] Next Sample",
            "  [Left] Previous Sample",
            "  [E] Toggle Error Map",
            "  [A] Toggle Auto Scale",
            "  [ESC] Exit",
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = font.render(line, True, TEXT_COLOR)
            screen.blit(text_surface, (self.margin, y + i * 20))
        
        # 显示当前样本的统计信息
        if self.current_sample_idx < len(self.samples):
            sample = self.samples[self.current_sample_idx]
            current_max = np.max(sample['current'])
            pred_max = np.max(sample['predicted'])
            true_max = np.max(sample['true'])
            error_mean = np.mean(sample['error'])
            error_max = np.max(sample['error'])
            
            stats_x = self.panel_width * 3 + self.margin * 2
            stats_lines = [
                "Statistics:",
                f"Current Max Density: {current_max:.3f}",
                f"Predicted Max Density: {pred_max:.3f}",
                f"True Max Density: {true_max:.3f}",
                f"Mean Error: {error_mean:.4f}",
                f"Max Error: {error_max:.4f}",
            ]
            
            for i, line in enumerate(stats_lines):
                text_surface = font.render(line, True, TEXT_COLOR)
                screen.blit(text_surface, (stats_x, y + i * 20))
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """绘制场景"""
        screen.fill(BG_COLOR)
        
        if self.current_sample_idx >= len(self.samples):
            return
        
        sample = self.samples[self.current_sample_idx]
        
        # 计算最大值用于自动缩放
        max_val = 1.0
        if self.auto_scale:
            max_current = np.max(sample['current'])
            max_pred = np.max(sample['predicted'])
            max_true = np.max(sample['true'])
            # 取三者最大值，并设个下限避免除以零或过度放大噪声
            max_val = max(0.1, max(max_current, max_pred, max_true))
        
        # 计算面板位置
        panel_x = self.margin
        panel_y = self.margin + 25
        
        if self.show_error:
            # 显示误差图模式
            self._draw_error_field(
                screen,
                sample['error'],
                panel_x,
                panel_y,
                "Prediction Error",
                font,
            )
            
            self._draw_density_field(
                screen,
                sample['predicted'],
                panel_x + self.panel_width + self.margin,
                panel_y,
                "Predicted",
                font,
                max_val,
            )
            
            self._draw_density_field(
                screen,
                sample['true'],
                panel_x + (self.panel_width + self.margin) * 2,
                panel_y,
                "Ground Truth",
                font,
                max_val,
            )
        else:
            # 显示密度场模式
            self._draw_density_field(
                screen,
                sample['current'],
                panel_x,
                panel_y,
                "Current (Input)",
                font,
                max_val,
            )
            
            self._draw_density_field(
                screen,
                sample['predicted'],
                panel_x + self.panel_width + self.margin,
                panel_y,
                "Predicted (5s)",
                font,
                max_val,
            )
            
            self._draw_density_field(
                screen,
                sample['true'],
                panel_x + (self.panel_width + self.margin) * 2,
                panel_y,
                "Ground Truth (5s)",
                font,
                max_val,
            )
        
        # 绘制信息面板
        self._draw_info_panel(screen, font)
    
    def handle_event(self, event: pygame.event.Event):
        """处理事件"""
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_RIGHT:
                self.current_sample_idx = (self.current_sample_idx + 1) % len(self.samples)
            elif event.key == pygame.K_LEFT:
                self.current_sample_idx = (self.current_sample_idx - 1) % len(self.samples)
            elif event.key == pygame.K_e:
                self.show_error = not self.show_error
            elif event.key == pygame.K_a:
                self.auto_scale = not self.auto_scale
    
    def run(self):
        """运行可视化"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Density Field Prediction Visualization")
        clock = pygame.time.Clock()
        
        # 使用系统默认字体而不是中文字体，避免编码问题
        font = pygame.font.Font(None, 14)
        
        print("\n" + "=" * 60)
        print("Density Field Prediction Visualization")
        print("=" * 60)
        print("Controls:")
        print("  [Space] Pause/Play")
        print("  [Right Arrow] Next Sample")
        print("  [Left Arrow] Previous Sample")
        print("  [E] Toggle Error Map")
        print("  [ESC] Exit")
        print("=" * 60 + "\n")
        
        frame_count = 0
        
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            
            # 自动播放
            if not self.paused:
                if frame_count % 30 == 0:  # 每30帧切换一次（约0.5秒）
                    self.current_sample_idx = (self.current_sample_idx + 1) % len(self.samples)
                frame_count += 1
            
            self.draw(screen, font)
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        print("\nVisualization closed")


def main():
    parser = argparse.ArgumentParser(description="Density Field Prediction Visualization")
    parser.add_argument("--model-path", type=str, default="outputs/models/density_predictor.pt",
                        help="Model path")
    parser.add_argument("--data-dir", type=str, default="outputs/training_data",
                        help="Data directory")
    parser.add_argument("--lite", action="store_true", help="Use lite model")
    parser.add_argument("--cell-size", type=int, default=20, help="Pixel size for each grid cell")
    
    args = parser.parse_args()
    
    visualizer = DensityFieldVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        use_lite_model=args.lite,
        cell_size=args.cell_size,
    )
    
    visualizer.run()


if __name__ == "__main__":
    main()
