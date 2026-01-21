"""
Trajectron++ Trajectory Prediction Model

A graph-based multi-modal trajectory prediction model inspired by:
- Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data
  (Salzmann et al., ECCV 2020)

Key Features:
- Graph Neural Network for agent-agent interactions
- Multi-modal prediction (outputs multiple possible trajectories)
- GMM (Gaussian Mixture Model) decoder for uncertainty estimation
- Support for scene context (optional)

Architecture:
1. Node History Encoder (LSTM): Encodes individual agent history
2. Edge Encoder (Attention): Captures agent-agent interactions
3. Scene Encoder (CNN, optional): Encodes scene map
4. CVAE Decoder: Generates diverse predictions

Simplified implementation for crowd evacuation scenario.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path


class NodeHistoryEncoder(nn.Module):
    """Encodes individual agent's trajectory history using LSTM

    Input: Historical positions (obs_len, 2)
    Output: Encoded state (hidden_dim,)
    """

    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Position embedding
        self.pos_embedding = nn.Linear(input_dim, embedding_dim)

        # LSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (batch, obs_len, 2) historical positions

        Returns:
            encoded: (batch, hidden_dim) encoded state
        """
        # Embed positions
        embedded = self.pos_embedding(history)  # (batch, obs_len, embedding_dim)
        embedded = self.dropout(embedded)

        # LSTM encoding
        _, (h_n, _) = self.lstm(embedded)

        # Return last layer's hidden state
        return h_n[-1]  # (batch, hidden_dim)


class EdgeEncoder(nn.Module):
    """Encodes agent-agent interactions using attention mechanism

    Uses multi-head attention to aggregate information from neighbors
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_neighbors: int = 10
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors

        # Relative position encoder
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        seq_start_end: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Args:
            node_features: (total_peds, hidden_dim) node hidden states
            positions: (total_peds, 2) current positions
            seq_start_end: List of (start, end) indices for each scene

        Returns:
            edge_features: (total_peds, hidden_dim) aggregated neighbor features
        """
        batch_size = node_features.size(0)
        device = node_features.device

        edge_features = torch.zeros(batch_size, self.hidden_dim, device=device)

        for start, end in seq_start_end:
            num_peds = end - start
            if num_peds <= 1:
                continue

            # Get node features and positions for this scene
            scene_features = node_features[start:end]  # (num_peds, hidden_dim)
            scene_positions = positions[start:end]  # (num_peds, 2)

            # Compute relative positions (each agent relative to all others)
            rel_positions = (
                scene_positions.unsqueeze(0) -
                scene_positions.unsqueeze(1)
            )  # (num_peds, num_peds, 2)

            # Encode relative positions
            rel_encoded = self.rel_pos_encoder(rel_positions)  # (num_peds, num_peds, hidden_dim)

            # Combine with node features for attention
            # Query: current agent, Key/Value: other agents
            query = scene_features.unsqueeze(1)  # (num_peds, 1, hidden_dim)
            key = scene_features.unsqueeze(0).expand(num_peds, -1, -1)  # (num_peds, num_peds, hidden_dim)
            key = key + rel_encoded  # Add position encoding
            value = key

            # Self-attention (exclude self using key_padding_mask alternative)
            # Note: We don't mask self since the model can learn to handle it
            attn_output, _ = self.attention(
                query, key, value
            )  # (num_peds, 1, hidden_dim)

            attn_output = attn_output.squeeze(1)  # (num_peds, hidden_dim)

            # Combine with original features
            combined = torch.cat([scene_features, attn_output], dim=-1)
            edge_features[start:end] = self.output_proj(combined)

        return edge_features


class GMMDecoder(nn.Module):
    """Gaussian Mixture Model Decoder for multi-modal trajectory prediction

    Outputs K possible future trajectories with associated probabilities
    """

    def __init__(
        self,
        input_dim: int = 256,
        pred_len: int = 12,
        num_modes: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 2
    ):
        super().__init__()
        self.pred_len = pred_len
        self.num_modes = num_modes
        self.output_dim = output_dim

        # Mode selection network
        self.mode_probs = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes),
            nn.Softmax(dim=-1)
        )

        # Trajectory decoder for each mode
        self.trajectory_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, pred_len * output_dim)
            )
            for _ in range(num_modes)
        ])

        # Optional: Uncertainty estimation (sigma)
        self.uncertainty_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, pred_len * output_dim),
                nn.Softplus()  # Ensure positive variance
            )
            for _ in range(num_modes)
        ])

    def forward(
        self,
        features: torch.Tensor,
        last_pos: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, input_dim) encoded features
            last_pos: (batch, 2) last observed position (for relative prediction)

        Returns:
            predictions: (batch, num_modes, pred_len, 2) predicted trajectories
            probs: (batch, num_modes) mode probabilities
            sigmas: (batch, num_modes, pred_len, 2) prediction uncertainties
        """
        batch_size = features.size(0)

        # Get mode probabilities
        probs = self.mode_probs(features)  # (batch, num_modes)

        # Decode trajectories for each mode
        predictions = []
        sigmas = []

        for i in range(self.num_modes):
            # Trajectory prediction (relative displacements)
            traj = self.trajectory_decoders[i](features)
            traj = traj.view(batch_size, self.pred_len, self.output_dim)
            predictions.append(traj)

            # Uncertainty
            sigma = self.uncertainty_decoders[i](features)
            sigma = sigma.view(batch_size, self.pred_len, self.output_dim)
            sigmas.append(sigma)

        predictions = torch.stack(predictions, dim=1)  # (batch, num_modes, pred_len, 2)
        sigmas = torch.stack(sigmas, dim=1)  # (batch, num_modes, pred_len, 2)

        # Convert relative to absolute positions
        if last_pos is not None:
            # Cumulative sum to get absolute positions
            predictions_cumsum = torch.cumsum(predictions, dim=2)
            predictions = predictions_cumsum + last_pos.unsqueeze(1).unsqueeze(2)

        return predictions, probs, sigmas


class TrajectronPlusPlus(nn.Module):
    """Trajectron++ Trajectory Prediction Model

    Full architecture:
    1. NodeHistoryEncoder: LSTM for individual history
    2. EdgeEncoder: Attention for agent interactions
    3. GMMDecoder: Multi-modal trajectory output
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_modes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_encoder: bool = True
    ):
        """
        Args:
            obs_len: Observation sequence length
            pred_len: Prediction sequence length
            input_dim: Input dimension (2 for x,y)
            embedding_dim: Position embedding dimension
            hidden_dim: LSTM and attention hidden dimension
            num_modes: Number of prediction modes
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_edge_encoder: Whether to use agent-agent interactions
        """
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.use_edge_encoder = use_edge_encoder

        # Node history encoder
        self.node_encoder = NodeHistoryEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Edge encoder (optional)
        if use_edge_encoder:
            self.edge_encoder = EdgeEncoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            decoder_input_dim = hidden_dim * 2
        else:
            self.edge_encoder = None
            decoder_input_dim = hidden_dim

        # GMM decoder
        self.decoder = GMMDecoder(
            input_dim=decoder_input_dim,
            pred_len=pred_len,
            num_modes=num_modes,
            hidden_dim=hidden_dim
        )

    def forward(
        self,
        obs_traj: torch.Tensor,
        seq_start_end: List[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_traj: (obs_len, total_peds, 2) observed trajectories
            seq_start_end: List of (start, end) indices for each scene

        Returns:
            predictions: (total_peds, num_modes, pred_len, 2) multi-modal predictions
            probs: (total_peds, num_modes) mode probabilities
        """
        # Transpose to (total_peds, obs_len, 2)
        obs_traj = obs_traj.permute(1, 0, 2)
        total_peds = obs_traj.size(0)

        if seq_start_end is None:
            seq_start_end = [(0, total_peds)]

        # Encode node histories
        node_features = self.node_encoder(obs_traj)  # (total_peds, hidden_dim)

        # Encode edges (agent interactions)
        if self.use_edge_encoder and self.edge_encoder is not None:
            last_positions = obs_traj[:, -1, :]  # (total_peds, 2)
            edge_features = self.edge_encoder(
                node_features, last_positions, seq_start_end
            )
            # Combine node and edge features
            combined_features = torch.cat([node_features, edge_features], dim=-1)
        else:
            combined_features = node_features

        # Decode predictions
        last_pos = obs_traj[:, -1, :]  # (total_peds, 2)
        predictions, probs, sigmas = self.decoder(combined_features, last_pos)

        return predictions, probs

    def predict(
        self,
        obs_traj: torch.Tensor,
        seq_start_end: List[Tuple[int, int]] = None,
        mode: str = 'best'
    ) -> torch.Tensor:
        """Predict future trajectories

        Args:
            obs_traj: (obs_len, total_peds, 2) observed trajectories
            seq_start_end: Scene indices
            mode: 'best' (most likely), 'sample' (weighted sample), or 'all'

        Returns:
            predictions: (pred_len, total_peds, 2) for best/sample
                        or (total_peds, num_modes, pred_len, 2) for all
        """
        predictions, probs = self.forward(obs_traj, seq_start_end)

        if mode == 'all':
            return predictions  # (total_peds, num_modes, pred_len, 2)
        elif mode == 'sample':
            # Sample mode based on probabilities
            batch_size = predictions.size(0)
            mode_indices = torch.multinomial(probs, 1).squeeze(-1)
            sampled = predictions[
                torch.arange(batch_size),
                mode_indices
            ]  # (total_peds, pred_len, 2)
            return sampled.permute(1, 0, 2)  # (pred_len, total_peds, 2)
        else:  # 'best'
            # Select most likely mode
            best_modes = probs.argmax(dim=-1)
            batch_size = predictions.size(0)
            best_pred = predictions[
                torch.arange(batch_size),
                best_modes
            ]  # (total_peds, pred_len, 2)
            return best_pred.permute(1, 0, 2)  # (pred_len, total_peds, 2)

    @classmethod
    def load(cls, model_path: str, device: str = 'cpu') -> 'TrajectronPlusPlus':
        """Load trained model from checkpoint

        Args:
            model_path: Path to model checkpoint
            device: Device to load model on

        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Create model with saved config
        config = checkpoint.get('config', {})
        model = cls(
            obs_len=config.get('obs_len', 8),
            pred_len=config.get('pred_len', 12),
            embedding_dim=config.get('embedding_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            num_modes=config.get('num_modes', 5),
            num_heads=config.get('num_heads', 4),
            dropout=0.0,  # No dropout during inference
            use_edge_encoder=config.get('use_edge_encoder', True)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"Trajectron++ model loaded: {model_path}")
        print(f"  obs_len={model.obs_len}, pred_len={model.pred_len}, "
              f"num_modes={model.num_modes}")

        return model

    def save(self, model_path: str):
        """Save model checkpoint

        Args:
            model_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'obs_len': self.obs_len,
                'pred_len': self.pred_len,
                'hidden_dim': self.hidden_dim,
                'num_modes': self.num_modes,
                'use_edge_encoder': self.use_edge_encoder,
            }
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to: {model_path}")


class TrajectronLoss(nn.Module):
    """Loss function for Trajectron++ training

    Combines:
    - Negative log-likelihood for GMM
    - ADE/FDE regularization
    - Mode diversity encouragement
    """

    def __init__(
        self,
        num_modes: int = 5,
        mode_diversity_weight: float = 0.1,
        min_ade_weight: float = 1.0
    ):
        super().__init__()
        self.num_modes = num_modes
        self.mode_diversity_weight = mode_diversity_weight
        self.min_ade_weight = min_ade_weight

    def forward(
        self,
        predictions: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: (batch, num_modes, pred_len, 2) predicted trajectories
            probs: (batch, num_modes) mode probabilities
            targets: (pred_len, batch, 2) ground truth trajectories

        Returns:
            loss: Total loss
            metrics: Dictionary of individual loss components
        """
        batch_size = predictions.size(0)
        num_modes = predictions.size(1)
        pred_len = predictions.size(2)

        # Transpose targets to (batch, pred_len, 2)
        targets = targets.permute(1, 0, 2)

        # Compute ADE for each mode
        # predictions: (batch, num_modes, pred_len, 2)
        # targets: (batch, 1, pred_len, 2)
        targets_expanded = targets.unsqueeze(1)
        errors = predictions - targets_expanded  # (batch, num_modes, pred_len, 2)
        distances = torch.norm(errors, dim=-1)  # (batch, num_modes, pred_len)

        # ADE per mode: (batch, num_modes)
        ade_per_mode = distances.mean(dim=-1)

        # FDE per mode: (batch, num_modes)
        fde_per_mode = distances[:, :, -1]

        # Min ADE loss (encourage at least one good mode)
        min_ade, min_idx = ade_per_mode.min(dim=1)
        min_ade_loss = min_ade.mean()

        # Min FDE loss
        min_fde = fde_per_mode.min(dim=1)[0].mean()

        # Weighted ADE loss (weight by mode probabilities)
        weighted_ade = (ade_per_mode * probs).sum(dim=1).mean()

        # Mode diversity loss (encourage diverse predictions)
        # Compute pairwise distances between modes
        mode_diffs = predictions.unsqueeze(2) - predictions.unsqueeze(1)
        mode_distances = torch.norm(mode_diffs, dim=-1).mean(dim=-1)  # (batch, num_modes, num_modes)
        diversity_loss = -mode_distances.mean()  # Maximize diversity

        # Total loss
        loss = (
            self.min_ade_weight * min_ade_loss +
            weighted_ade +
            self.mode_diversity_weight * diversity_loss
        )

        metrics = {
            'loss': loss.item(),
            'min_ade': min_ade_loss.item(),
            'min_fde': min_fde.item(),
            'weighted_ade': weighted_ade.item(),
            'diversity': -diversity_loss.item()
        }

        return loss, metrics


def compute_multimodal_metrics(
    predictions: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """Compute evaluation metrics for multi-modal predictions

    Args:
        predictions: (batch, num_modes, pred_len, 2)
        probs: (batch, num_modes)
        targets: (pred_len, batch, 2)

    Returns:
        metrics: Dictionary with minADE, minFDE, avgADE, etc.
    """
    # Transpose targets
    targets = targets.permute(1, 0, 2)  # (batch, pred_len, 2)
    targets_expanded = targets.unsqueeze(1)  # (batch, 1, pred_len, 2)

    # Compute errors
    errors = predictions - targets_expanded
    distances = torch.norm(errors, dim=-1)  # (batch, num_modes, pred_len)

    # ADE per mode
    ade_per_mode = distances.mean(dim=-1)  # (batch, num_modes)

    # FDE per mode
    fde_per_mode = distances[:, :, -1]  # (batch, num_modes)

    # Min ADE/FDE (best mode)
    min_ade = ade_per_mode.min(dim=1)[0].mean().item()
    min_fde = fde_per_mode.min(dim=1)[0].mean().item()

    # Best mode ADE (most probable)
    best_modes = probs.argmax(dim=1)
    batch_indices = torch.arange(predictions.size(0))
    best_ade = ade_per_mode[batch_indices, best_modes].mean().item()
    best_fde = fde_per_mode[batch_indices, best_modes].mean().item()

    # Weighted average
    weighted_ade = (ade_per_mode * probs).sum(dim=1).mean().item()
    weighted_fde = (fde_per_mode * probs).sum(dim=1).mean().item()

    return {
        'minADE': min_ade,
        'minFDE': min_fde,
        'bestADE': best_ade,
        'bestFDE': best_fde,
        'weightedADE': weighted_ade,
        'weightedFDE': weighted_fde,
    }


if __name__ == '__main__':
    # Test the model
    print("Testing Trajectron++ model...")

    # Create model
    model = TrajectronPlusPlus(
        obs_len=8,
        pred_len=12,
        hidden_dim=128,
        num_modes=5
    )

    # Test forward pass
    batch_size = 16
    obs_len = 8
    pred_len = 12

    obs_traj = torch.randn(obs_len, batch_size, 2)
    targets = torch.randn(pred_len, batch_size, 2)
    seq_start_end = [(0, batch_size)]

    predictions, probs = model(obs_traj, seq_start_end)

    print(f"\nInput shape: {obs_traj.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs.sum(dim=1)}")

    # Test loss
    loss_fn = TrajectronLoss()
    loss, metrics = loss_fn(predictions, probs, targets)

    print(f"\nLoss: {loss.item():.4f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test predict method
    best_pred = model.predict(obs_traj, seq_start_end, mode='best')
    print(f"\nBest prediction shape: {best_pred.shape}")

    # Test save/load
    model.save('/tmp/test_trajectron.pt')
    loaded_model = TrajectronPlusPlus.load('/tmp/test_trajectron.pt')
    print("\nModel save/load test passed!")
