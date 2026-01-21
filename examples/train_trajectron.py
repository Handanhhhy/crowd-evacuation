#!/usr/bin/env python3
"""
Trajectron++ Trajectory Prediction Model Training Script

Trains the Trajectron++ model on pedestrian trajectory data for
multi-modal trajectory prediction in crowd evacuation scenarios.

Features:
- Multi-modal prediction (5 possible futures)
- Graph-based agent interactions
- Supports multiple datasets (ETH/UCY, SDD, Grand Central Station)
- GPU acceleration with CUDA support

Usage:
    python examples/train_trajectron.py
    python examples/train_trajectron.py --config configs/trajectron_config.yaml
    python examples/train_trajectron.py --epochs 100 --batch_size 32

Model saved to:
    outputs/models/trajectron.pt
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict
import yaml
from tqdm import tqdm

from ml.trajectron import (
    TrajectronPlusPlus,
    TrajectronLoss,
    compute_multimodal_metrics
)
from ml.data_loader import (
    TrajectoryDataset,
    MultiDatasetLoader,
    collate_trajectories,
    compute_dataset_statistics
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Trajectron++ trajectory prediction model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (YAML)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(project_root / 'data' / 'raw'),
        help='Path to data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(project_root / 'outputs' / 'models'),
        help='Path to output directory'
    )
    parser.add_argument(
        '--obs_len',
        type=int,
        default=8,
        help='Observation sequence length'
    )
    parser.add_argument(
        '--pred_len',
        type=int,
        default=12,
        help='Prediction sequence length'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Hidden dimension size'
    )
    parser.add_argument(
        '--num_modes',
        type=int,
        default=5,
        help='Number of prediction modes'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation'
    )
    parser.add_argument(
        '--no_edge_encoder',
        action='store_true',
        help='Disable edge encoder (agent interactions)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (cpu, cuda, or auto)'
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_arg: str) -> torch.device:
    """Get appropriate device"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS can be unstable, use CPU for reliability
            device = torch.device('cpu')
            print("MPS available but using CPU for stability")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"Using device: {device}")
    return device


def train_epoch(
    model: TrajectronPlusPlus,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TrajectronLoss,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_metrics = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        obs_traj = batch['obs_traj'].to(device)
        pred_traj = batch['pred_traj'].to(device)
        seq_start_end = batch['seq_start_end']

        optimizer.zero_grad()

        # Forward pass
        predictions, probs = model(obs_traj, seq_start_end)

        # Compute loss
        loss, metrics = loss_fn(predictions, probs, pred_traj)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Accumulate metrics
        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0) + value
        num_batches += 1

        pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})

    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= max(num_batches, 1)

    return total_metrics


def evaluate(
    model: TrajectronPlusPlus,
    dataloader: DataLoader,
    loss_fn: TrajectronLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    total_metrics = {}
    eval_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            obs_traj = batch['obs_traj'].to(device)
            pred_traj = batch['pred_traj'].to(device)
            seq_start_end = batch['seq_start_end']

            # Forward pass
            predictions, probs = model(obs_traj, seq_start_end)

            # Compute loss
            _, metrics = loss_fn(predictions, probs, pred_traj)

            # Compute evaluation metrics
            eval_m = compute_multimodal_metrics(predictions, probs, pred_traj)

            # Accumulate
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + value
            for key, value in eval_m.items():
                eval_metrics[key] = eval_metrics.get(key, 0) + value

            num_batches += 1

    # Average
    for key in total_metrics:
        total_metrics[key] /= max(num_batches, 1)
    for key in eval_metrics:
        eval_metrics[key] /= max(num_batches, 1)

    total_metrics.update(eval_metrics)
    return total_metrics


def main():
    args = parse_args()

    print("=" * 60)
    print("Trajectron++ Trajectory Prediction Model Training")
    print("=" * 60)

    # Load config file if provided
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        print(f"\nLoaded config from: {args.config}")
        # Override args with config
        for key, value in config.get('model', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
        for key, value in config.get('training', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Print configuration
    print("\nConfiguration:")
    print(f"  obs_len: {args.obs_len}")
    print(f"  pred_len: {args.pred_len}")
    print(f"  hidden_dim: {args.hidden_dim}")
    print(f"  num_modes: {args.num_modes}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  dropout: {args.dropout}")
    print(f"  augment: {args.augment}")
    print(f"  use_edge_encoder: {not args.no_edge_encoder}")

    # Get device
    device = get_device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    data_loader = MultiDatasetLoader(
        data_root=args.data_dir,
        datasets=['eth_ucy'],  # Can add 'sdd', 'grand_central' if available
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        augment=args.augment,
        batch_size=args.batch_size
    )

    train_loader = data_loader.create_dataloader('train')
    val_loader = data_loader.create_dataloader('val')

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Compute dataset statistics
    print("\nDataset statistics:")
    stats = compute_dataset_statistics(train_loader)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = TrajectronPlusPlus(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        hidden_dim=args.hidden_dim,
        num_modes=args.num_modes,
        num_heads=4,
        dropout=args.dropout,
        use_edge_encoder=not args.no_edge_encoder
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    loss_fn = TrajectronLoss(
        num_modes=args.num_modes,
        mode_diversity_weight=0.1,
        min_ade_weight=1.0
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_min_ade = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = 20

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        # Learning rate scheduling
        scheduler.step(val_metrics['minADE'])

        # Print metrics
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"minADE: {train_metrics['min_ade']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"minADE: {val_metrics['minADE']:.4f}m, "
              f"minFDE: {val_metrics['minFDE']:.4f}m, "
              f"bestADE: {val_metrics['bestADE']:.4f}m")

        # Save best model
        if val_metrics['minADE'] < best_min_ade:
            best_min_ade = val_metrics['minADE']
            best_epoch = epoch
            patience_counter = 0

            model_path = output_dir / 'trajectron.pt'
            model.save(str(model_path))
            print(f"  -> New best model saved (minADE: {best_min_ade:.4f}m)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best minADE: {best_min_ade:.4f}m")
    print(f"Model saved to: {output_dir / 'trajectron.pt'}")

    # Load best model and final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model = TrajectronPlusPlus.load(str(output_dir / 'trajectron.pt'), device=str(device))
    final_metrics = evaluate(model, val_loader, loss_fn, device)

    print("\nFinal metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Compare with Social-LSTM baseline if available
    lstm_model_path = output_dir / 'social_lstm.pt'
    if lstm_model_path.exists():
        print("\n" + "=" * 60)
        print("Comparison with Social-LSTM baseline")
        print("=" * 60)
        print("(Run examples/train_trajectory.py to train Social-LSTM)")
        print(f"  Trajectron++ minADE: {best_min_ade:.4f}m")
        # Note: Would need to load and evaluate Social-LSTM for fair comparison

    # Test prediction
    print("\n" + "=" * 60)
    print("Test Prediction")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            obs_traj = batch['obs_traj'].to(device)
            pred_traj = batch['pred_traj'].to(device)
            seq_start_end = batch['seq_start_end']

            # Multi-modal prediction
            predictions, probs = model(obs_traj, seq_start_end)

            print(f"\nObservation shape: {obs_traj.shape}")
            print(f"Ground truth shape: {pred_traj.shape}")
            print(f"Predictions shape: {predictions.shape}")
            print(f"Mode probabilities: {probs[0].cpu().numpy()}")

            # Best prediction
            best_pred = model.predict(obs_traj, seq_start_end, mode='best')
            print(f"Best prediction shape: {best_pred.shape}")

            # Sample trajectory for visualization
            sample_idx = 0
            print(f"\nSample trajectory (pedestrian {sample_idx}):")
            print(f"  Last observed position: {obs_traj[-1, sample_idx].cpu().numpy()}")
            print(f"  Ground truth final: {pred_traj[-1, sample_idx].cpu().numpy()}")
            for mode in range(min(3, args.num_modes)):
                pred_final = predictions[sample_idx, mode, -1].cpu().numpy()
                prob = probs[sample_idx, mode].item()
                print(f"  Mode {mode} (p={prob:.3f}) final: {pred_final}")

            break

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
