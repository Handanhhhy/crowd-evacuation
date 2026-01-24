#!/usr/bin/env python
"""
密度场预测模型训练脚本

步骤：
1. 运行SFM仿真收集数据（多个episode）
2. 构建序列数据集
3. 训练ConvLSTM
4. 保存模型到 outputs/models/density_predictor.pt

使用方法:
    # 收集数据并训练
    python scripts/train_density_predictor.py --collect-data --train
    
    # 仅收集数据
    python scripts/train_density_predictor.py --collect-data --n-episodes 20
    
    # 使用已有数据训练
    python scripts/train_density_predictor.py --train --epochs 100
    
    # 评估模型
    python scripts/train_density_predictor.py --evaluate --model-path outputs/models/density_predictor.pt

参考文档: docs/new_station_plan.md 密度场预测模块 TODO
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

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
    MAX_SAFE_DENSITY,
)
from prediction.data_collector import create_dataloader
from simulation.large_station_env import LargeStationEnv


def collect_training_data(
    n_episodes: int = 10,
    flow_level: str = "small",
    max_steps: int = 3000,
    save_dir: str = "outputs/training_data",
    dt: float = 0.1,
    collect_interval: int = 1,
) -> DensityDataCollector:
    """收集训练数据
    
    Args:
        n_episodes: episode数量
        flow_level: 人流量等级 (small/medium/large)
        max_steps: 每个episode最大步数
        save_dir: 数据保存目录
        dt: 仿真时间步长
        collect_interval: 数据收集间隔（每N步收集一次）
        
    Returns:
        DensityDataCollector: 数据收集器
    """
    print("=" * 60)
    print("收集训练数据")
    print("=" * 60)
    print(f"配置:")
    print(f"  - Episodes: {n_episodes}")
    print(f"  - 人流等级: {flow_level}")
    print(f"  - 最大步数: {max_steps}")
    print(f"  - 收集间隔: 每{collect_interval}步")
    print()
    
    # 创建环境
    env = LargeStationEnv(
        flow_level=flow_level,
        max_steps=max_steps,
        dt=dt,
        emergency_mode=True,
    )
    
    # 提取出口信息
    exits = [{'id': e.id, 'position': e.position.copy()} for e in env.exits]
    
    # 创建数据收集器
    collector = DensityDataCollector(
        exits=exits,
        save_dir=save_dir,
    )
    
    for episode_idx in range(n_episodes):
        print(f"\n[Episode {episode_idx + 1}/{n_episodes}]")
        
        collector.start_episode({
            'episode': episode_idx,
            'flow_level': flow_level,
            'max_steps': max_steps,
            'dt': dt,
        })
        
        obs, _ = env.reset()
        step = 0
        
        pbar = tqdm(total=max_steps, desc=f"Episode {episode_idx + 1}")
        
        while step < max_steps:
            # 随机动作（用于数据收集，不需要策略）
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 收集数据
            if step % collect_interval == 0:
                # 获取行人数据
                pedestrians = []
                for ped in env.sfm.pedestrians:
                    pedestrians.append({
                        'position': ped.position.copy(),
                        'velocity': ped.velocity.copy(),
                    })
                
                timestamp = step * dt
                collector.collect_frame(pedestrians, timestamp)
            
            step += 1
            pbar.update(1)
            
            if terminated:
                break
        
        pbar.close()
        
        # 保存episode
        episode_name = f"episode_{episode_idx:04d}_{flow_level}"
        collector.save_episode(episode_name)
        collector.end_episode()
        
        print(f"  疏散: {info.get('evacuated', 0)}人")
        print(f"  剩余: {info.get('remaining', 0)}人")
        print(f"  收集帧数: {len(collector.episodes[-1]) if collector.episodes else 0}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("数据收集完成")
    stats = collector.get_statistics()
    print(f"  - Episodes: {stats['n_episodes']}")
    print(f"  - 总帧数: {stats['total_frames']}")
    print("=" * 60)
    
    return collector


def train_model(
    data_dir: str = "outputs/training_data",
    model_save_path: str = "outputs/models/density_predictor.pt",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seq_length: int = 10,
    pred_horizon: int = 50,
    use_lite_model: bool = False,
    device: str = "auto",
) -> nn.Module:
    """训练密度预测模型
    
    Args:
        data_dir: 训练数据目录
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        seq_length: 输入序列长度（10帧 = 1秒）
        pred_horizon: 预测步长（50帧 = 5秒）
        use_lite_model: 是否使用轻量级模型
        device: 训练设备
        
    Returns:
        训练好的模型
    """
    print("=" * 60)
    print("训练密度预测模型")
    print("=" * 60)
    
    # 设备检测和打印
    print("\n设备信息:")
    print(f"  - PyTorch版本: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n  - 使用设备: {device}")
    if device == "cuda":
        print(f"  - 当前GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 加载数据
    # 首先创建一个临时的collector来获取exits信息
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"训练数据目录不存在: {data_dir}")
    
    # 从第一个episode获取出口信息
    first_episode = None
    for ep_dir in sorted(data_path.iterdir()):
        if ep_dir.is_dir() and (ep_dir / "frames.pkl").exists():
            first_episode = ep_dir
            break
    
    if first_episode is None:
        raise FileNotFoundError(f"未找到有效的训练数据")
    
    # 读取第一帧获取出口距离场
    import pickle
    with open(first_episode / "frames.pkl", 'rb') as f:
        frames_data = pickle.load(f)
    
    # 创建虚拟exits列表（仅用于初始化collector）
    exits = [{'id': f'exit_{i}', 'position': np.array([0, 0])} for i in range(8)]
    
    collector = DensityDataCollector(
        exits=exits,
        save_dir=data_dir,
    )
    collector.load_all_episodes()
    
    # 构建数据集
    print(f"\n构建数据集...")
    print(f"  - 序列长度: {seq_length}帧 ({seq_length * 0.1}秒)")
    print(f"  - 预测步长: {pred_horizon}帧 ({pred_horizon * 0.1}秒)")
    
    train_dataset, val_dataset = collector.build_dataset(
        seq_length=seq_length,
        pred_horizon=pred_horizon,
        stride=5,
        train_ratio=0.8,
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print(f"\n创建模型...")
    if use_lite_model:
        model = DensityPredictorLite(
            input_channels=4,
            hidden_channels=32,
            grid_size=GRID_SIZE,
        )
        print("  - 模型类型: Lite")
    else:
        model = DensityPredictorNet(
            input_channels=4,
            hidden_channels=64,
            encoder_channels=32,
            num_lstm_layers=2,
            grid_size=GRID_SIZE,
        )
        print("  - 模型类型: Full")
    
    model = model.to(device)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 可训练参数: {n_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    
    # 训练记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
    }
    
    # 训练循环
    print(f"\n开始训练 ({epochs} epochs)...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            pred, _ = model(batch_x)
            loss = criterion(pred, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                pred, _ = model(batch_x)
                loss = criterion(pred, batch_y)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch + 1
            
            # 保存模型
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  [*] 保存最佳模型 (val_loss: {avg_val_loss:.6f})")
        
        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
    
    # 保存训练历史
    history_path = Path(model_save_path).with_suffix('.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': history['best_val_loss'],
            'best_epoch': history['best_epoch'],
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'seq_length': seq_length,
                'pred_horizon': pred_horizon,
                'use_lite_model': use_lite_model,
            },
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成")
    print(f"  - 最佳验证损失: {history['best_val_loss']:.6f} (Epoch {history['best_epoch']})")
    print(f"  - 模型保存: {model_save_path}")
    print(f"  - 历史保存: {history_path}")
    print("=" * 60)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    return model


def evaluate_model(
    model_path: str = "outputs/models/density_predictor.pt",
    data_dir: str = "outputs/training_data",
    device: str = "auto",
    use_lite_model: bool = False,
):
    """评估模型性能
    
    Args:
        model_path: 模型路径
        data_dir: 测试数据目录
        device: 设备
        use_lite_model: 是否为轻量级模型
    """
    print("=" * 60)
    print("评估模型")
    print("=" * 60)
    
    # 设备检测和打印
    print("\n设备信息:")
    print(f"  - PyTorch版本: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n  - 使用设备: {device}")
    if device == "cuda":
        print(f"  - 当前GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 加载模型
    if use_lite_model:
        model = DensityPredictorLite(input_channels=4, hidden_channels=32, grid_size=GRID_SIZE)
    else:
        model = DensityPredictorNet(input_channels=4, hidden_channels=64, grid_size=GRID_SIZE)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"加载模型: {model_path}")
    
    # 加载数据
    exits = [{'id': f'exit_{i}', 'position': np.array([0, 0])} for i in range(8)]
    collector = DensityDataCollector(exits=exits, save_dir=data_dir)
    collector.load_all_episodes()
    
    _, test_dataset = collector.build_dataset(
        seq_length=10,
        pred_horizon=50,
        train_ratio=0.0,  # 全部用于测试
    )
    
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
    
    # 评估
    criterion = nn.MSELoss()
    losses = []
    mae_values = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred, _ = model(batch_x)
            
            loss = criterion(pred, batch_y)
            mae = torch.abs(pred - batch_y).mean()
            
            losses.append(loss.item())
            mae_values.append(mae.item())
    
    avg_mse = np.mean(losses)
    avg_mae = np.mean(mae_values)
    avg_rmse = np.sqrt(avg_mse)
    
    # 转换为实际密度值
    rmse_density = avg_rmse * MAX_SAFE_DENSITY
    mae_density = avg_mae * MAX_SAFE_DENSITY
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"  MSE (归一化): {avg_mse:.6f}")
    print(f"  RMSE (归一化): {avg_rmse:.6f}")
    print(f"  MAE (归一化): {avg_mae:.6f}")
    print(f"  RMSE (人/m²): {rmse_density:.4f}")
    print(f"  MAE (人/m²): {mae_density:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="密度场预测模型训练")
    
    # 操作选择
    parser.add_argument("--collect-data", action="store_true", help="收集训练数据")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    
    # 数据收集参数
    parser.add_argument("--n-episodes", type=int, default=10, help="收集的episode数量")
    parser.add_argument("--flow-level", type=str, default="small", 
                        choices=["small", "medium", "large"], help="人流量等级")
    parser.add_argument("--max-steps", type=int, default=3000, help="每个episode最大步数")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seq-length", type=int, default=10, help="输入序列长度")
    parser.add_argument("--pred-horizon", type=int, default=50, help="预测步长")
    parser.add_argument("--lite", action="store_true", help="使用轻量级模型")
    
    # 路径
    parser.add_argument("--data-dir", type=str, default="outputs/training_data", help="数据目录")
    parser.add_argument("--model-path", type=str, default="outputs/models/density_predictor.pt", 
                        help="模型路径")
    
    # 设备
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    
    args = parser.parse_args()
    
    # 在开始时打印设备信息
    print("\n" + "=" * 60)
    print("系统设备信息")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
    else:
        print("  (将使用CPU进行训练)")
    print("=" * 60 + "\n")
    
    # 默认行为：同时收集数据和训练
    if not args.collect_data and not args.train and not args.evaluate:
        args.collect_data = True
        args.train = True
    
    # 收集数据
    if args.collect_data:
        collect_training_data(
            n_episodes=args.n_episodes,
            flow_level=args.flow_level,
            max_steps=args.max_steps,
            save_dir=args.data_dir,
        )
    
    # 训练
    if args.train:
        train_model(
            data_dir=args.data_dir,
            model_save_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seq_length=args.seq_length,
            pred_horizon=args.pred_horizon,
            use_lite_model=args.lite,
            device=args.device,
        )
    
    # 评估
    if args.evaluate:
        evaluate_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            device=args.device,
            use_lite_model=args.lite,
        )


if __name__ == "__main__":
    main()
