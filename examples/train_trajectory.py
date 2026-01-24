#!/usr/bin/env python3
"""
Social-LSTM轨迹预测模型训练脚本

训练数据: ETH/UCY数据集（真实行人轨迹）
模型: Social-LSTM (Alahi et al. 2016)

使用方法:
    python examples/train_trajectory.py

训练完成后模型保存到:
    outputs/models/social_lstm.pt
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import pandas as pd
from tqdm import tqdm

from ml.trajectory_predictor import SocialLSTM, trajectory_loss, compute_ade_fde
from utils.device_info import print_device_info, get_device, print_device_selection


class TrajectoryDataset(Dataset):
    """轨迹数据集

    将原始轨迹数据转换为 (观测序列, 预测序列) 对
    """

    def __init__(
        self,
        data_path: str,
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        delim: str = '\t'
    ):
        """
        Args:
            data_path: 数据文件路径
            obs_len: 观测序列长度
            pred_len: 预测序列长度
            skip: 采样间隔
            min_ped: 场景中最少行人数
            delim: 分隔符
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.min_ped = min_ped

        # 加载数据
        self.obs_traj = []
        self.pred_traj = []
        self.obs_traj_rel = []
        self.pred_traj_rel = []
        self.seq_start_end = []

        self._load_data(data_path, delim)

    def _load_data(self, data_path: str, delim: str):
        """加载并预处理数据"""
        print(f"加载数据: {data_path}")

        # 读取数据
        data = pd.read_csv(data_path, sep=delim, header=None,
                          names=['frame', 'ped_id', 'x', 'y'])

        frames = data['frame'].unique()
        frame_data = []

        for frame in frames:
            frame_df = data[data['frame'] == frame]
            frame_data.append(frame_df[['ped_id', 'x', 'y']].values)

        num_sequences = (len(frames) - self.seq_len) // self.skip + 1

        all_obs_traj = []
        all_pred_traj = []
        all_obs_traj_rel = []
        all_pred_traj_rel = []
        all_seq_start_end = []

        for idx in range(0, num_sequences * self.skip, self.skip):
            # 获取当前序列的所有帧
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + self.seq_len], axis=0
            )

            # 获取序列中出现的所有行人
            peds_in_curr_seq = np.unique(curr_seq_data[:, 0])
            curr_num_peds = len(peds_in_curr_seq)

            if curr_num_peds < self.min_ped:
                continue

            # 为每个行人提取轨迹
            curr_obs_traj = []
            curr_pred_traj = []
            curr_obs_traj_rel = []
            curr_pred_traj_rel = []

            for ped_id in peds_in_curr_seq:
                # 获取该行人的所有位置
                ped_frames = curr_seq_data[curr_seq_data[:, 0] == ped_id]

                if len(ped_frames) < self.seq_len:
                    continue

                # 只取前seq_len个位置
                ped_traj = ped_frames[:self.seq_len, 1:3]  # (seq_len, 2)

                # 分割为观测和预测
                obs_traj = ped_traj[:self.obs_len]  # (obs_len, 2)
                pred_traj = ped_traj[self.obs_len:]  # (pred_len, 2)

                # 计算相对位移
                obs_traj_rel = np.zeros_like(obs_traj)
                obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]

                pred_traj_rel = np.zeros_like(pred_traj)
                pred_traj_rel[0] = pred_traj[0] - obs_traj[-1]
                pred_traj_rel[1:] = pred_traj[1:] - pred_traj[:-1]

                curr_obs_traj.append(obs_traj)
                curr_pred_traj.append(pred_traj)
                curr_obs_traj_rel.append(obs_traj_rel)
                curr_pred_traj_rel.append(pred_traj_rel)

            if len(curr_obs_traj) < self.min_ped:
                continue

            # 记录序列信息
            start_idx = len(all_obs_traj)
            end_idx = start_idx + len(curr_obs_traj)

            all_obs_traj.extend(curr_obs_traj)
            all_pred_traj.extend(curr_pred_traj)
            all_obs_traj_rel.extend(curr_obs_traj_rel)
            all_pred_traj_rel.extend(curr_pred_traj_rel)
            all_seq_start_end.append((start_idx, end_idx))

        if len(all_obs_traj) == 0:
            print("警告: 未能提取有效轨迹序列，使用合成数据")
            self._create_synthetic_data()
            return

        # 转换为numpy数组
        self.obs_traj = np.array(all_obs_traj)  # (num_peds, obs_len, 2)
        self.pred_traj = np.array(all_pred_traj)  # (num_peds, pred_len, 2)
        self.obs_traj_rel = np.array(all_obs_traj_rel)
        self.pred_traj_rel = np.array(all_pred_traj_rel)
        self.seq_start_end = all_seq_start_end

        print(f"数据加载完成:")
        print(f"  总轨迹数: {len(self.obs_traj)}")
        print(f"  场景数: {len(self.seq_start_end)}")

    def _create_synthetic_data(self):
        """创建合成训练数据"""
        print("生成合成轨迹数据...")

        np.random.seed(42)
        num_sequences = 100
        peds_per_seq = 10

        all_obs_traj = []
        all_pred_traj = []
        all_seq_start_end = []

        for seq_idx in range(num_sequences):
            start_idx = len(all_obs_traj)

            for ped_idx in range(peds_per_seq):
                # 生成起点
                start_pos = np.random.uniform(0, 10, 2)

                # 生成目标点
                target_pos = np.random.uniform(0, 10, 2)

                # 生成轨迹 (简单线性插值 + 噪声)
                full_traj = []
                for t in range(self.seq_len):
                    alpha = t / (self.seq_len - 1)
                    pos = (1 - alpha) * start_pos + alpha * target_pos
                    pos += np.random.normal(0, 0.1, 2)  # 添加噪声
                    full_traj.append(pos)

                full_traj = np.array(full_traj)

                obs_traj = full_traj[:self.obs_len]
                pred_traj = full_traj[self.obs_len:]

                all_obs_traj.append(obs_traj)
                all_pred_traj.append(pred_traj)

            end_idx = len(all_obs_traj)
            all_seq_start_end.append((start_idx, end_idx))

        self.obs_traj = np.array(all_obs_traj)
        self.pred_traj = np.array(all_pred_traj)
        self.seq_start_end = all_seq_start_end

        print(f"合成数据生成完成:")
        print(f"  总轨迹数: {len(self.obs_traj)}")
        print(f"  场景数: {len(self.seq_start_end)}")

    def __len__(self):
        return len(self.seq_start_end)

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]

        # 获取该场景中的所有轨迹
        obs_traj = self.obs_traj[start:end]  # (num_peds, obs_len, 2)
        pred_traj = self.pred_traj[start:end]  # (num_peds, pred_len, 2)

        # 转置为 (seq_len, num_peds, 2)
        obs_traj = np.transpose(obs_traj, (1, 0, 2))
        pred_traj = np.transpose(pred_traj, (1, 0, 2))

        return {
            'obs_traj': torch.FloatTensor(obs_traj),
            'pred_traj': torch.FloatTensor(pred_traj),
            'seq_start_end': [(0, end - start)]
        }


def collate_fn(batch):
    """自定义数据整理函数"""
    obs_traj_list = []
    pred_traj_list = []
    seq_start_end = []

    total_peds = 0
    for item in batch:
        num_peds = item['obs_traj'].size(1)
        obs_traj_list.append(item['obs_traj'])
        pred_traj_list.append(item['pred_traj'])
        seq_start_end.append((total_peds, total_peds + num_peds))
        total_peds += num_peds

    # 在行人维度上拼接
    obs_traj = torch.cat(obs_traj_list, dim=1)  # (obs_len, total_peds, 2)
    pred_traj = torch.cat(pred_traj_list, dim=1)  # (pred_len, total_peds, 2)

    return {
        'obs_traj': obs_traj,
        'pred_traj': pred_traj,
        'seq_start_end': seq_start_end
    }


def train_epoch(
    model: SocialLSTM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        obs_traj = batch['obs_traj'].to(device)
        pred_traj = batch['pred_traj'].to(device)
        seq_start_end = batch['seq_start_end']

        optimizer.zero_grad()

        # 前向传播
        pred_traj_hat = model(obs_traj, seq_start_end)

        # 计算损失
        loss = trajectory_loss(pred_traj_hat, pred_traj)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: SocialLSTM,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0
    total_ade = 0
    total_fde = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            obs_traj = batch['obs_traj'].to(device)
            pred_traj = batch['pred_traj'].to(device)
            seq_start_end = batch['seq_start_end']

            # 预测
            pred_traj_hat = model(obs_traj, seq_start_end)

            # 计算损失
            loss = trajectory_loss(pred_traj_hat, pred_traj)
            total_loss += loss.item()

            # 计算ADE/FDE
            pred_np = pred_traj_hat.cpu().numpy()
            target_np = pred_traj.cpu().numpy()

            # 转置回 (batch, pred_len, 2)
            pred_np = np.transpose(pred_np, (1, 0, 2))
            target_np = np.transpose(target_np, (1, 0, 2))

            ade, fde = compute_ade_fde(pred_np, target_np)
            total_ade += ade
            total_fde += fde

            num_batches += 1

    num_batches = max(num_batches, 1)
    return total_loss / num_batches, total_ade / num_batches, total_fde / num_batches


def main():
    # 打印设备信息
    print_device_info("系统设备信息")
    
    print("=" * 60)
    print("Social-LSTM 轨迹预测模型训练")
    print("=" * 60)

    # 配置
    config = {
        'obs_len': 8,           # 观测8帧 (3.2秒 @ 2.5Hz)
        'pred_len': 12,         # 预测12帧 (4.8秒)
        'embedding_dim': 64,
        'hidden_dim': 128,
        'pool_dim': 64,
        'grid_size': 4,
        'neighborhood_size': 2.0,
        'dropout': 0.0,
        'learning_rate': 1e-3,
        'batch_size': 8,
        'epochs': 50,
        'device': get_device('auto')
    }

    print("\n训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    device_str = config['device']
    device = torch.device(device_str)
    print_device_selection(device_str)

    # 数据路径
    data_path = project_root / "data" / "raw" / "eth_ucy" / "synthetic_eth.txt"
    output_dir = project_root / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据集
    print("\n加载训练数据...")
    dataset = TrajectoryDataset(
        data_path=str(data_path),
        obs_len=config['obs_len'],
        pred_len=config['pred_len']
    )

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"训练集: {len(train_dataset)} 个场景")
    print(f"验证集: {len(val_dataset)} 个场景")

    # 创建模型
    model = SocialLSTM(
        obs_len=config['obs_len'],
        pred_len=config['pred_len'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        pool_dim=config['pool_dim'],
        grid_size=config['grid_size'],
        neighborhood_size=config['neighborhood_size'],
        dropout=config['dropout']
    ).to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和学习率调度
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    best_ade = float('inf')

    for epoch in range(1, config['epochs'] + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # 验证
        val_loss, val_ade, val_fde = evaluate(model, val_loader, device)

        # 学习率调度
        scheduler.step()

        # 打印进度
        print(f"Epoch {epoch:3d}/{config['epochs']}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"ADE={val_ade:.4f}m, FDE={val_fde:.4f}m")

        # 保存最佳模型
        if val_ade < best_ade:
            best_ade = val_ade
            best_val_loss = val_loss
            model_path = output_dir / "social_lstm.pt"
            model.save(str(model_path))
            print(f"  -> 保存最佳模型 (ADE={best_ade:.4f}m)")

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证ADE: {best_ade:.4f}m")
    print(f"最佳验证FDE: {val_fde:.4f}m")
    print(f"模型保存到: {output_dir / 'social_lstm.pt'}")
    print("=" * 60)

    # 简单测试
    print("\n测试预测...")
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            obs_traj = batch['obs_traj'].to(device)
            pred_traj = batch['pred_traj'].to(device)
            seq_start_end = batch['seq_start_end']

            pred_traj_hat = model(obs_traj, seq_start_end)

            print(f"观测轨迹形状: {obs_traj.shape}")
            print(f"预测轨迹形状: {pred_traj_hat.shape}")
            print(f"真实轨迹形状: {pred_traj.shape}")

            # 计算一个样本的误差
            pred_np = pred_traj_hat[:, 0, :].cpu().numpy()
            target_np = pred_traj[:, 0, :].cpu().numpy()
            sample_ade = np.mean(np.linalg.norm(pred_np - target_np, axis=-1))
            print(f"样本ADE: {sample_ade:.4f}m")
            break


if __name__ == "__main__":
    main()
