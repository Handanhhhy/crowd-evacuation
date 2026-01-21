"""
Unified Trajectory Data Loader

Supports multiple public pedestrian trajectory datasets:
- ETH/UCY: Classic benchmark datasets (street scenes)
- Stanford Drone Dataset (SDD): Bird's eye view trajectories
- Grand Central Station: Train station scenario (most relevant)

Output format: PyTorch DataLoader with (obs_traj, pred_traj, neighbors)

Reference datasets:
- ETH/UCY: Pellegrini et al. 2009, Lerner et al. 2007
- SDD: Robicquet et al. 2016
- Grand Central: Yi et al. 2015
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd


class TrajectoryDataset(Dataset):
    """Unified Trajectory Dataset

    Loads trajectory data from various formats and provides
    consistent (obs_traj, pred_traj) pairs for training.

    Supports:
    - ETH/UCY format: frame_id, ped_id, x, y (tab-separated)
    - SDD format: similar structure with additional annotations
    - Grand Central: pixel coordinates (need scaling)
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        delim: str = '\t',
        scale: float = 1.0,
        augment: bool = False,
        dataset_type: str = 'eth_ucy'
    ):
        """
        Args:
            data_path: Path to data file(s). Can be single file or list
            obs_len: Observation sequence length (frames)
            pred_len: Prediction sequence length (frames)
            skip: Sampling interval
            min_ped: Minimum pedestrians per scene
            delim: Delimiter for CSV/text files
            scale: Coordinate scaling factor (for pixel->meter conversion)
            augment: Whether to apply data augmentation
            dataset_type: 'eth_ucy', 'sdd', or 'grand_central'
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.min_ped = min_ped
        self.scale = scale
        self.augment = augment
        self.dataset_type = dataset_type

        # Data storage
        self.obs_traj = []      # (num_seqs, num_peds, obs_len, 2)
        self.pred_traj = []     # (num_seqs, num_peds, pred_len, 2)
        self.obs_traj_rel = []  # Relative displacements
        self.pred_traj_rel = []
        self.seq_start_end = [] # (start, end) indices for each scene
        self.num_peds_per_seq = []

        # Load data
        if isinstance(data_path, (str, Path)):
            data_path = [data_path]

        for path in data_path:
            self._load_file(Path(path), delim)

        # Convert to numpy arrays
        if len(self.obs_traj) > 0:
            self.obs_traj = np.array(self.obs_traj)
            self.pred_traj = np.array(self.pred_traj)
            if len(self.obs_traj_rel) > 0:
                self.obs_traj_rel = np.array(self.obs_traj_rel)
                self.pred_traj_rel = np.array(self.pred_traj_rel)

    def _load_file(self, file_path: Path, delim: str):
        """Load a single data file"""
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            return

        print(f"Loading: {file_path}")

        if self.dataset_type == 'eth_ucy':
            self._load_eth_ucy(file_path, delim)
        elif self.dataset_type == 'sdd':
            self._load_sdd(file_path)
        elif self.dataset_type == 'grand_central':
            self._load_grand_central(file_path)
        else:
            # Default to ETH/UCY format
            self._load_eth_ucy(file_path, delim)

    def _load_eth_ucy(self, file_path: Path, delim: str):
        """Load ETH/UCY format data

        Format: frame_id, ped_id, x, y (tab or space separated)
        """
        # Read data
        data = pd.read_csv(
            file_path,
            sep=delim,
            header=None,
            names=['frame', 'ped_id', 'x', 'y']
        )

        # Apply scaling
        data['x'] = data['x'] * self.scale
        data['y'] = data['y'] * self.scale

        # Get unique frames
        frames = data['frame'].unique()
        frames.sort()

        # Group by frame
        frame_data = []
        for frame in frames:
            frame_df = data[data['frame'] == frame]
            frame_data.append(frame_df[['ped_id', 'x', 'y']].values)

        # Extract sequences
        num_sequences = (len(frames) - self.seq_len) // self.skip + 1

        for idx in range(0, num_sequences * self.skip, self.skip):
            if idx + self.seq_len > len(frame_data):
                break

            # Get current sequence frames
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + self.seq_len], axis=0
            )

            # Get unique pedestrians in this sequence
            peds_in_seq = np.unique(curr_seq_data[:, 0])

            if len(peds_in_seq) < self.min_ped:
                continue

            # Extract trajectories for each pedestrian
            seq_obs = []
            seq_pred = []
            seq_obs_rel = []
            seq_pred_rel = []

            for ped_id in peds_in_seq:
                ped_frames = curr_seq_data[curr_seq_data[:, 0] == ped_id]

                if len(ped_frames) < self.seq_len:
                    continue

                # Get trajectory
                ped_traj = ped_frames[:self.seq_len, 1:3]  # (seq_len, 2)

                # Split into obs and pred
                obs_traj = ped_traj[:self.obs_len]  # (obs_len, 2)
                pred_traj = ped_traj[self.obs_len:]  # (pred_len, 2)

                # Compute relative displacements
                obs_traj_rel = np.zeros_like(obs_traj)
                obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]

                pred_traj_rel = np.zeros_like(pred_traj)
                pred_traj_rel[0] = pred_traj[0] - obs_traj[-1]
                pred_traj_rel[1:] = pred_traj[1:] - pred_traj[:-1]

                seq_obs.append(obs_traj)
                seq_pred.append(pred_traj)
                seq_obs_rel.append(obs_traj_rel)
                seq_pred_rel.append(pred_traj_rel)

            if len(seq_obs) < self.min_ped:
                continue

            # Store sequence
            self.obs_traj.extend(seq_obs)
            self.pred_traj.extend(seq_pred)
            self.obs_traj_rel.extend(seq_obs_rel)
            self.pred_traj_rel.extend(seq_pred_rel)

            start_idx = len(self.obs_traj) - len(seq_obs)
            end_idx = len(self.obs_traj)
            self.seq_start_end.append((start_idx, end_idx))
            self.num_peds_per_seq.append(len(seq_obs))

        print(f"  Loaded {len(self.seq_start_end)} sequences, "
              f"{len(self.obs_traj)} trajectories")

    def _load_sdd(self, file_path: Path):
        """Load Stanford Drone Dataset format

        SDD format: trackId, xmin, ymin, xmax, ymax, frame, lost, occluded,
                    generated, label
        We use center of bounding box as position.
        """
        data = pd.read_csv(
            file_path,
            sep=' ',
            header=None,
            names=['track_id', 'xmin', 'ymin', 'xmax', 'ymax',
                   'frame', 'lost', 'occluded', 'generated', 'label']
        )

        # Filter out lost and occluded
        data = data[(data['lost'] == 0) & (data['occluded'] == 0)]

        # Compute center position
        data['x'] = (data['xmin'] + data['xmax']) / 2 * self.scale
        data['y'] = (data['ymin'] + data['ymax']) / 2 * self.scale

        # Convert to ETH format
        eth_data = pd.DataFrame({
            'frame': data['frame'],
            'ped_id': data['track_id'],
            'x': data['x'],
            'y': data['y']
        })

        # Save temporarily and load with ETH loader
        temp_path = file_path.parent / f"{file_path.stem}_converted.txt"
        eth_data.to_csv(temp_path, sep='\t', header=False, index=False)
        self._load_eth_ucy(temp_path, '\t')
        temp_path.unlink()  # Delete temp file

    def _load_grand_central(self, file_path: Path):
        """Load Grand Central Station dataset

        Format varies, typically: ped_id, frame, x, y (pixel coordinates)
        Needs scaling from pixels to meters (approx 0.01 m/pixel)
        """
        # Grand Central uses different format
        # Adjust scale for pixel->meter conversion
        original_scale = self.scale
        self.scale = self.scale * 0.01  # Typical pixel to meter ratio

        # Try loading as ETH format
        try:
            self._load_eth_ucy(file_path, '\t')
        except Exception:
            # Try space-separated
            self._load_eth_ucy(file_path, ' ')

        self.scale = original_scale

    def apply_augmentation(
        self,
        obs: np.ndarray,
        pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation

        Augmentations:
        - Random rotation
        - Random scaling
        - Random flipping
        - Velocity noise
        """
        if not self.augment:
            return obs, pred

        # Combine for joint transformation
        traj = np.concatenate([obs, pred], axis=0)  # (seq_len, 2)

        # Random rotation (+-30 degrees)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-np.pi/6, np.pi/6)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            traj = traj @ rotation_matrix.T

        # Random scaling (0.8x to 1.2x)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            traj = traj * scale

        # Random horizontal flip
        if np.random.random() < 0.5:
            traj[:, 0] = -traj[:, 0]

        # Random vertical flip
        if np.random.random() < 0.5:
            traj[:, 1] = -traj[:, 1]

        # Velocity noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, traj.shape)
            traj = traj + noise

        obs_aug = traj[:self.obs_len]
        pred_aug = traj[self.obs_len:]

        return obs_aug, pred_aug

    def __len__(self) -> int:
        return len(self.seq_start_end)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start, end = self.seq_start_end[idx]

        # Get trajectories for this scene
        obs_traj = self.obs_traj[start:end].copy()  # (num_peds, obs_len, 2)
        pred_traj = self.pred_traj[start:end].copy()  # (num_peds, pred_len, 2)

        # Apply augmentation
        if self.augment:
            for i in range(len(obs_traj)):
                obs_traj[i], pred_traj[i] = self.apply_augmentation(
                    obs_traj[i], pred_traj[i]
                )

        # Transpose to (seq_len, num_peds, 2)
        obs_traj = np.transpose(obs_traj, (1, 0, 2))
        pred_traj = np.transpose(pred_traj, (1, 0, 2))

        # Compute neighbor information (relative positions)
        num_peds = obs_traj.shape[1]
        last_obs = obs_traj[-1]  # (num_peds, 2)

        # Pairwise distances at last observation
        neighbor_info = np.zeros((num_peds, num_peds, 2))
        for i in range(num_peds):
            for j in range(num_peds):
                if i != j:
                    neighbor_info[i, j] = last_obs[j] - last_obs[i]

        return {
            'obs_traj': torch.FloatTensor(obs_traj),
            'pred_traj': torch.FloatTensor(pred_traj),
            'seq_start_end': [(0, end - start)],
            'neighbors': torch.FloatTensor(neighbor_info),
            'num_peds': num_peds
        }


def collate_trajectories(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for trajectory batches

    Concatenates trajectories from multiple scenes along the pedestrian dimension.
    """
    obs_traj_list = []
    pred_traj_list = []
    seq_start_end = []
    neighbor_list = []

    total_peds = 0

    for item in batch:
        num_peds = item['obs_traj'].size(1)
        obs_traj_list.append(item['obs_traj'])
        pred_traj_list.append(item['pred_traj'])
        seq_start_end.append((total_peds, total_peds + num_peds))
        neighbor_list.append(item['neighbors'])
        total_peds += num_peds

    # Concatenate along pedestrian dimension
    obs_traj = torch.cat(obs_traj_list, dim=1)  # (obs_len, total_peds, 2)
    pred_traj = torch.cat(pred_traj_list, dim=1)  # (pred_len, total_peds, 2)

    return {
        'obs_traj': obs_traj,
        'pred_traj': pred_traj,
        'seq_start_end': seq_start_end,
        'neighbors': neighbor_list,
        'total_peds': total_peds
    }


class MultiDatasetLoader:
    """Load and combine multiple trajectory datasets

    Supports loading from:
    - ETH (2 scenes: ETH, Hotel)
    - UCY (3 scenes: Zara1, Zara2, Univ)
    - SDD (multiple scenes)
    - Grand Central Station
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        datasets: List[str] = None,
        obs_len: int = 8,
        pred_len: int = 12,
        augment: bool = False,
        batch_size: int = 64,
        shuffle: bool = True
    ):
        """
        Args:
            data_root: Root directory containing dataset folders
            datasets: List of datasets to load ('eth', 'ucy', 'sdd', 'grand_central')
            obs_len: Observation sequence length
            pred_len: Prediction sequence length
            augment: Whether to apply data augmentation
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
        """
        self.data_root = Path(data_root)
        self.datasets = datasets or ['eth_ucy']
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.augment = augment
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Dataset paths (relative to data_root)
        self.dataset_paths = {
            'eth_ucy': [
                'eth_ucy/synthetic_eth.txt',
                'eth_ucy/eth.txt',
                'eth_ucy/hotel.txt',
                'eth_ucy/zara1.txt',
                'eth_ucy/zara2.txt',
                'eth_ucy/univ.txt',
            ],
            'sdd': [
                'sdd/bookstore_0.txt',
                'sdd/bookstore_1.txt',
                'sdd/coupa_0.txt',
                'sdd/coupa_1.txt',
                'sdd/deathCircle_0.txt',
                'sdd/gates_0.txt',
                'sdd/hyang_0.txt',
                'sdd/nexus_0.txt',
            ],
            'grand_central': [
                'grand_central/gc_trajectories.txt',
            ]
        }

    def get_available_files(self) -> List[Path]:
        """Get list of available data files"""
        available = []
        for dataset in self.datasets:
            if dataset in self.dataset_paths:
                for rel_path in self.dataset_paths[dataset]:
                    full_path = self.data_root / rel_path
                    if full_path.exists():
                        available.append(full_path)
        return available

    def create_dataloader(
        self,
        split: str = 'train',
        train_ratio: float = 0.8
    ) -> DataLoader:
        """Create DataLoader for training or validation

        Args:
            split: 'train' or 'val'
            train_ratio: Ratio of data to use for training

        Returns:
            DataLoader instance
        """
        available_files = self.get_available_files()

        if len(available_files) == 0:
            print("Warning: No data files found, creating synthetic data")
            # Create synthetic dataset
            dataset = self._create_synthetic_dataset()
        else:
            # Load from files
            dataset = TrajectoryDataset(
                data_path=available_files,
                obs_len=self.obs_len,
                pred_len=self.pred_len,
                augment=(self.augment and split == 'train')
            )

        # Split dataset
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        if split == 'train':
            data = train_dataset
        else:
            data = val_dataset

        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=(self.shuffle and split == 'train'),
            collate_fn=collate_trajectories,
            num_workers=0,
            drop_last=(split == 'train')
        )

    def _create_synthetic_dataset(self) -> TrajectoryDataset:
        """Create synthetic trajectory data for testing"""
        print("Generating synthetic trajectory data...")

        np.random.seed(42)
        num_scenes = 200
        peds_per_scene = np.random.randint(5, 15, num_scenes)

        all_data = []
        frame_id = 0
        ped_id = 0

        for scene_idx in range(num_scenes):
            num_peds = peds_per_scene[scene_idx]

            for _ in range(num_peds):
                # Random start and target
                start = np.random.uniform(0, 20, 2)
                target = np.random.uniform(0, 20, 2)

                # Generate trajectory with some noise
                for t in range(self.obs_len + self.pred_len):
                    alpha = t / (self.obs_len + self.pred_len - 1)
                    pos = (1 - alpha) * start + alpha * target
                    pos += np.random.normal(0, 0.1, 2)

                    all_data.append([frame_id + t, ped_id, pos[0], pos[1]])

                ped_id += 1

            frame_id += self.obs_len + self.pred_len + 10

        # Create temporary file
        df = pd.DataFrame(all_data, columns=['frame', 'ped_id', 'x', 'y'])
        temp_path = Path('/tmp/synthetic_traj.txt')
        df.to_csv(temp_path, sep='\t', header=False, index=False)

        dataset = TrajectoryDataset(
            data_path=temp_path,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            augment=self.augment
        )

        temp_path.unlink()
        return dataset


def compute_dataset_statistics(dataloader: DataLoader) -> Dict[str, float]:
    """Compute statistics of a trajectory dataset

    Returns:
        Dictionary with statistics (mean speed, acceleration, etc.)
    """
    all_speeds = []
    all_accelerations = []
    all_displacements = []

    for batch in dataloader:
        obs_traj = batch['obs_traj'].numpy()  # (obs_len, total_peds, 2)
        pred_traj = batch['pred_traj'].numpy()  # (pred_len, total_peds, 2)

        # Compute velocities
        full_traj = np.concatenate([obs_traj, pred_traj], axis=0)
        velocities = np.diff(full_traj, axis=0)
        speeds = np.linalg.norm(velocities, axis=-1)

        # Compute accelerations
        accelerations = np.diff(velocities, axis=0)
        acc_magnitudes = np.linalg.norm(accelerations, axis=-1)

        # Compute total displacement
        displacements = np.linalg.norm(
            pred_traj[-1] - obs_traj[0], axis=-1
        )

        all_speeds.extend(speeds.flatten())
        all_accelerations.extend(acc_magnitudes.flatten())
        all_displacements.extend(displacements.flatten())

    return {
        'mean_speed': np.mean(all_speeds),
        'std_speed': np.std(all_speeds),
        'max_speed': np.max(all_speeds),
        'mean_acceleration': np.mean(all_accelerations),
        'std_acceleration': np.std(all_accelerations),
        'mean_displacement': np.mean(all_displacements),
        'std_displacement': np.std(all_displacements),
    }


if __name__ == '__main__':
    # Test the data loader
    print("Testing TrajectoryDataset...")

    # Test with synthetic data
    loader = MultiDatasetLoader(
        data_root=Path(__file__).parent.parent.parent / 'data' / 'raw',
        datasets=['eth_ucy'],
        obs_len=8,
        pred_len=12,
        augment=True,
        batch_size=8
    )

    train_loader = loader.create_dataloader('train')
    val_loader = loader.create_dataloader('val')

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    for batch in train_loader:
        print(f"\nBatch info:")
        print(f"  obs_traj shape: {batch['obs_traj'].shape}")
        print(f"  pred_traj shape: {batch['pred_traj'].shape}")
        print(f"  seq_start_end: {batch['seq_start_end']}")
        print(f"  total_peds: {batch['total_peds']}")
        break

    # Compute statistics
    print("\nComputing dataset statistics...")
    stats = compute_dataset_statistics(train_loader)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
