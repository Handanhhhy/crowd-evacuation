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

# 密度颜色映射（从蓝色到红色）
def density_to_color(density, max_density=1.0):
    """将密度值映射到颜色（蓝色=低密度，红色=高密度）"""
    density = np.clip(density / max_density, 0.0, 1.0)
    
    if density < 0.5:
        # 蓝色到青色
        r = 0
        g = int(255 * density * 2)
        b = 255
    else:
        # 青色到黄色到红色
        r = int(255 * (density - 0.5) * 2)
        g = 255
        b = int(255 * (1 - (density - 0.5) * 2))
    
    return (r, g, b)


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
    ):
        """绘制密度场
        
        Args:
            screen: Pygame surface
            density: 密度场 [grid_w, grid_h]
            x_offset: X偏移
            y_offset: Y偏移
            title: 标题
        """
        # 绘制标题
        font = pygame.font.SysFont('Arial', 16)
        title_surface = font.render(title, True, TEXT_COLOR)
        screen.blit(title_surface, (x_offset + self.panel_width // 2 - title_surface.get_width() // 2, y_offset - 25))
        
        # 绘制网格
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                x = x_offset + i * self.cell_size
                y = y_offset + j * self.cell_size
                
                # 获取密度值
                d = density[i, j]
                color = density_to_color(d)
                
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
    ):
        """绘制误差场"""
        # 绘制标题
        font = pygame.font.SysFont('Arial', 16)
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
            f"样本: {self.current_sample_idx + 1} / {len(self.samples)}",
            f"状态: {'暂停' if self.paused else '播放'}",
            f"显示模式: {'误差图' if self.show_error else '密度场'}",
            "",
            "控制:",
            "  [空格] 暂停/播放",
            "  [→] 下一帧",
            "  [←] 上一帧",
            "  [E] 切换误差图",
            "  [ESC] 退出",
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
                "统计信息:",
                f"当前密度 (最大): {current_max:.3f}",
                f"预测密度 (最大): {pred_max:.3f}",
                f"真实密度 (最大): {true_max:.3f}",
                f"平均误差: {error_mean:.4f}",
                f"最大误差: {error_max:.4f}",
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
                "预测误差",
            )
            
            self._draw_density_field(
                screen,
                sample['predicted'],
                panel_x + self.panel_width + self.margin,
                panel_y,
                "预测密度",
            )
            
            self._draw_density_field(
                screen,
                sample['true'],
                panel_x + (self.panel_width + self.margin) * 2,
                panel_y,
                "真实密度",
            )
        else:
            # 显示密度场模式
            self._draw_density_field(
                screen,
                sample['current'],
                panel_x,
                panel_y,
                "当前密度 (输入)",
            )
            
            self._draw_density_field(
                screen,
                sample['predicted'],
                panel_x + self.panel_width + self.margin,
                panel_y,
                "预测密度 (5秒后)",
            )
            
            self._draw_density_field(
                screen,
                sample['true'],
                panel_x + (self.panel_width + self.margin) * 2,
                panel_y,
                "真实密度 (5秒后)",
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
    
    def run(self):
        """运行可视化"""
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("密度场预测可视化")
        clock = pygame.time.Clock()
        
        try:
            font = pygame.font.SysFont('Arial', 14)
        except:
            font = pygame.font.Font(None, 14)
        
        print("\n" + "=" * 60)
        print("密度场预测可视化")
        print("=" * 60)
        print("控制:")
        print("  [空格] 暂停/播放")
        print("  [→] 下一帧")
        print("  [←] 上一帧")
        print("  [E] 切换误差图")
        print("  [ESC] 退出")
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
        print("\n可视化已退出")


def main():
    parser = argparse.ArgumentParser(description="密度场预测可视化")
    parser.add_argument("--model-path", type=str, default="outputs/models/density_predictor.pt",
                        help="模型路径")
    parser.add_argument("--data-dir", type=str, default="outputs/training_data",
                        help="数据目录")
    parser.add_argument("--lite", action="store_true", help="使用轻量级模型")
    parser.add_argument("--cell-size", type=int, default=20, help="每个网格的像素大小")
    
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
