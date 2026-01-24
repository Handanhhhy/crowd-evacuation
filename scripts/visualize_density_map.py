#!/usr/bin/env python
"""
密度场地图可视化工具

在真实场景地图上叠加密度预测热力图。
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
    CELL_SIZE,
    SCENE_SIZE,
    MAX_SAFE_DENSITY,
)
from prediction.data_collector import create_dataloader
from simulation.large_station_env import LargeStationEnv

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (128, 128, 128)
LIGHT_GRAY = (220, 220, 220)
BG_COLOR = (245, 245, 245)
TEXT_COLOR = (50, 50, 50)
WALL_COLOR = (80, 80, 80)
EXIT_COLOR = (0, 200, 0)
ESCALATOR_COLOR = (200, 200, 0)

# 密度颜色映射（热力图风格）
def density_to_color(density, max_val=1.0, alpha=128):
    """将密度值映射到热力图颜色"""
    val = np.clip(density / max_val, 0.0, 1.0)
    
    # 颜色点 (值, (R, G, B))
    colors = [
        (0.0, (0, 0, 255)),    # 蓝 (低)
        (0.3, (0, 255, 255)),  # 青
        (0.5, (0, 255, 0)),    # 绿
        (0.7, (255, 255, 0)),  # 黄
        (1.0, (255, 0, 0))     # 红 (高)
    ]
    
    color = colors[-1][1]
    for i in range(len(colors) - 1):
        if val <= colors[i+1][0]:
            c1 = colors[i]
            c2 = colors[i+1]
            ratio = (val - c1[0]) / (c2[0] - c1[0])
            r = int(c1[1][0] + ratio * (c2[1][0] - c1[1][0]))
            g = int(c1[1][1] + ratio * (c2[1][1] - c1[1][1]))
            b = int(c1[1][2] + ratio * (c2[1][2] - c1[1][2]))
            color = (r, g, b)
            break
            
    return (*color, int(alpha * val))  # Alpha 随密度增加


class DensityMapVisualizer:
    """密度场地图可视化器"""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = "outputs/training_data",
        scale: float = 8.0,  # 像素/米
        margin: int = 50,
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.scale = scale
        self.margin = margin
        
        # 场景尺寸
        self.scene_w, self.scene_h = SCENE_SIZE
        self.grid_w, self.grid_h = GRID_SIZE
        
        # 窗口尺寸
        self.map_width = int(self.scene_w * scale)
        self.map_height = int(self.scene_h * scale)
        self.window_width = self.map_width + margin * 2
        self.window_height = self.map_height + margin * 2 + 100
        
        # 加载环境（获取结构信息）
        self.env = LargeStationEnv(flow_level="small")
        self.env.reset()  # 必须调用 reset 才能创建 sfm 并添加墙壁
        
        # 兼容不同版本的 SFM (CPU/GPU)
        self.walls = []
        if hasattr(self.env.sfm, 'walls'):
            self.walls = self.env.sfm.walls
        elif hasattr(self.env.sfm, 'obstacles'): # GPU版可能叫 obstacles
            self.walls = self.env.sfm.obstacles
            
        # 确保墙壁数据是numpy数组列表
        if self.walls and isinstance(self.walls[0], torch.Tensor):
            self.walls = [w.cpu().numpy() for w in self.walls]
            
        # 加载模型和数据
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.test_loader = self._load_data()
        
        # 状态
        self.samples = []
        self.current_idx = 0
        self.paused = False
        self.running = True
        self.show_prediction = True  # True=显示预测, False=显示真实
        self.auto_scale = True
        
        self._load_samples()
        
    def _load_model(self):
        print(f"[Visualizer] 加载模型: {self.model_path}")
        model = DensityPredictorNet(
            input_channels=4,
            hidden_channels=64,
            grid_size=GRID_SIZE,
        )
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_data(self):
        print(f"[Visualizer] 加载数据: {self.data_dir}")
        exits = [{'id': f'exit_{i}', 'position': np.array([0, 0])} for i in range(8)]
        collector = DensityDataCollector(exits=exits, save_dir=self.data_dir)
        collector.load_all_episodes()
        _, test_dataset = collector.build_dataset(seq_length=10, pred_horizon=10, train_ratio=0.0)
        return create_dataloader(test_dataset, batch_size=1, shuffle=False)
    
    def _load_samples(self):
        print("[Visualizer] 加载样本...")
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.test_loader, desc="处理数据"):
                batch_x = batch_x.to(self.device)
                pred, _ = self.model(batch_x)
                
                self.samples.append({
                    'current': batch_x[0, -1, 0, :, :].cpu().numpy(),
                    'predicted': pred[0, 0, :, :].cpu().numpy(),
                    'true': batch_y[0, 0, :, :].cpu().numpy(),
                })
        print(f"[Visualizer] 已加载 {len(self.samples)} 个样本")

    def world_to_screen(self, pos):
        """世界坐标 -> 屏幕坐标"""
        x = self.margin + int(pos[0] * self.scale)
        y = self.margin + int((self.scene_h - pos[1]) * self.scale)
        return (x, y)
    
    def draw_map(self, screen):
        """绘制地图背景"""
        # 绘制墙壁
        for wall in self.walls:
            start = self.world_to_screen(wall[0])
            end = self.world_to_screen(wall[1])
            pygame.draw.line(screen, WALL_COLOR, start, end, 3)
            
        # 绘制出口
        for exit in self.env.exits:
            pos = self.world_to_screen(exit.position)
            width = int(exit.width * self.scale)
            # 简化绘制为绿色矩形
            rect = pygame.Rect(0, 0, width, 10)
            rect.center = pos
            pygame.draw.rect(screen, EXIT_COLOR, rect)
            
        # 绘制扶梯区域
        for esc in self.env.escalators:
            pos = self.world_to_screen(esc.position)
            w = int(esc.size[0] * self.scale)
            h = int(esc.size[1] * self.scale)
            rect = pygame.Rect(0, 0, w, h)
            rect.center = pos
            pygame.draw.rect(screen, ESCALATOR_COLOR, rect, 2)

    def draw_heatmap(self, screen, density, max_val=1.0):
        """绘制密度热力图"""
        surface = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
        
        cell_w = self.map_width / self.grid_w
        cell_h = self.map_height / self.grid_h
        
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                val = density[i, j]
                if val > 0.05:  # 忽略极低密度
                    color = density_to_color(val, max_val, alpha=180)
                    
                    # 注意：网格坐标 (i, j) 对应世界坐标 x, y
                    # 但屏幕坐标 y 是翻转的
                    # 网格 (0,0) 是左下角，屏幕 (0, map_h) 是左下角
                    x = i * cell_w
                    y = self.map_height - (j + 1) * cell_h
                    
                    rect = pygame.Rect(x, y, cell_w + 1, cell_h + 1)
                    pygame.draw.rect(surface, color, rect)
        
        screen.blit(surface, (self.margin, self.margin))

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Density Field Map Visualization")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self.running = False
                    elif event.key == pygame.K_SPACE: self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT: self.current_idx = (self.current_idx + 1) % len(self.samples)
                    elif event.key == pygame.K_LEFT: self.current_idx = (self.current_idx - 1) % len(self.samples)
                    elif event.key == pygame.K_m: self.show_prediction = not self.show_prediction
                    elif event.key == pygame.K_a: self.auto_scale = not self.auto_scale

            screen.fill(BG_COLOR)
            self.draw_map(screen)
            
            if self.samples:
                sample = self.samples[self.current_idx]
                data = sample['predicted'] if self.show_prediction else sample['true']
                
                max_val = 1.0
                if self.auto_scale:
                    max_val = max(0.1, np.max(data))
                
                self.draw_heatmap(screen, data, max_val)
                
                # 信息显示
                info = [
                    f"Sample: {self.current_idx}/{len(self.samples)}",
                    f"Mode: {'Predicted' if self.show_prediction else 'Ground Truth'}",
                    f"Max Density: {np.max(data):.2f}",
                    f"Auto Scale: {'ON' if self.auto_scale else 'OFF'}",
                    "[M] Switch Mode  [Space] Pause  [A] Auto Scale"
                ]
                
                for i, line in enumerate(info):
                    text = font.render(line, True, TEXT_COLOR)
                    screen.blit(text, (self.margin, self.window_height - 100 + i * 25))

            if not self.paused:
                self.current_idx = (self.current_idx + 1) % len(self.samples)
                
            pygame.display.flip()
            clock.tick(30)
            
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/models/density_predictor.pt")
    parser.add_argument("--data-dir", default="outputs/training_data/large")
    args = parser.parse_args()
    
    viz = DensityMapVisualizer(args.model_path, args.data_dir)
    viz.run()