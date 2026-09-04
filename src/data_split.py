"""
Dataset splitting and loading with video-identity-level splits to prevent leakage.
Supports FaceForensics++, Celeb-DF, and DFDC datasets.
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


class DatasetSplitter:
    """
    Splits datasets at video/identity level to prevent train-test leakage.
    Handles multiple manipulation types: Deepfakes, Face2Face, FaceSwap, NeuralTextures.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.splits = {}

    def split_by_video_identity(
        self,
        data_root: str,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        dataset_type: str = "faceforensics++"
    ) -> Dict[str, List[str]]:
        """
        Split dataset at video/identity level, not frame level.

        Args:
            data_root: Root directory containing frames organized by category
            test_ratio: Proportion for test set
            val_ratio: Proportion for validation set
            dataset_type: Type of dataset (faceforensics++, celeb-df, dfdc)

        Returns:
            Dictionary with 'train', 'val', 'test' splits, each containing frame paths
        """
        video_groups = self._group_frames_by_video(data_root)

        # Separate real and fake videos
        real_videos = []
        fake_videos = []

        for video_id, frames in video_groups.items():
            # Determine if real or fake based on frames
            if self._is_real_video(frames[0]):
                real_videos.append((video_id, frames))
            else:
                fake_videos.append((video_id, frames))

        print(f"Total videos: {len(real_videos) + len(fake_videos)}")
        print(f"  Real videos: {len(real_videos)}")
        print(f"  Fake videos: {len(fake_videos)}")

        # Split each category
        real_split = self._split_video_list(real_videos, test_ratio, val_ratio)
        fake_split = self._split_video_list(fake_videos, test_ratio, val_ratio)

        # Combine splits
        self.splits = {
            'train': real_split['train'] + fake_split['train'],
            'val': real_split['val'] + fake_split['val'],
            'test': real_split['test'] + fake_split['test']
        }

        # Shuffle each split
        for split_name in self.splits:
            random.shuffle(self.splits[split_name])

        # Log statistics
        self._log_split_stats()

        return self.splits

    def _group_frames_by_video(self, data_root: str) -> Dict[str, List[str]]:
        """Group frames by their source video."""
        video_groups = defaultdict(list)

        for root, dirs, files in os.walk(data_root):
            for file in sorted(files):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    # Extract video ID from filename (e.g., "fake1.mp4_0.jpg" -> "fake1.mp4")
                    video_id = '_'.join(file.split('_')[:-1])
                    frame_path = os.path.join(root, file)
                    video_groups[video_id].append(frame_path)

        return video_groups

    def _is_real_video(self, frame_path: str) -> bool:
        """Determine if frame belongs to real or fake video based on directory."""
        return "real" in frame_path.lower()

    def _split_video_list(
        self,
        videos: List[Tuple[str, List[str]]],
        test_ratio: float,
        val_ratio: float
    ) -> Dict[str, List[str]]:
        """Split a list of videos into train/val/test."""
        n_videos = len(videos)
        n_test = max(1, int(n_videos * test_ratio))
        n_val = max(1, int(n_videos * val_ratio))
        n_train = n_videos - n_test - n_val

        # Shuffle and split
        random.shuffle(videos)
        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]

        # Flatten to frames
        train_frames = self._flatten_frames(train_videos)
        val_frames = self._flatten_frames(val_videos)
        test_frames = self._flatten_frames(test_videos)

        return {
            'train': train_frames,
            'val': val_frames,
            'test': test_frames
        }

    def _flatten_frames(self, videos: List[Tuple[str, List[str]]]) -> List[str]:
        """Flatten list of (video_id, frames) tuples to just frames."""
        frames = []
        for video_id, frame_list in videos:
            frames.extend(frame_list)
        return frames

    def _log_split_stats(self):
        """Log split statistics."""
        total_frames = sum(len(frames) for frames in self.splits.values())

        print("\nDataset Split Statistics:")
        print(f"Seed: {self.seed}")
        for split_name, frames in self.splits.items():
            percentage = 100 * len(frames) / total_frames if total_frames > 0 else 0
            print(f"  {split_name}: {len(frames)} frames ({percentage:.1f}%)")
        print(f"Total frames: {total_frames}")

    def save_split(self, output_path: str):
        """Save split configuration to JSON."""
        split_config = {
            'seed': self.seed,
            'split_sizes': {
                split: len(frames) for split, frames in self.splits.items()
            },
            'splits': {
                split: [str(frame) for frame in frames]
                for split, frames in self.splits.items()
            }
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(split_config, f, indent=2)
        print(f"\nSplit configuration saved to {output_path}")

    def load_split(self, config_path: str):
        """Load split configuration from JSON."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.seed = config['seed']
        self.splits = {
            split: frames for split, frames in config['splits'].items()
        }
        print(f"Split configuration loaded from {config_path}")
        self._log_split_stats()


def get_split_summary(data_root: str, split_config_path: str = None) -> Dict:
    """
    Get summary of dataset splits without loading all data.

    Returns:
        Dictionary with split statistics
    """
    splitter = DatasetSplitter()

    if split_config_path and os.path.exists(split_config_path):
        splitter.load_split(split_config_path)
        splits = splitter.splits
    else:
        splits = splitter.split_by_video_identity(data_root)

    summary = {
        'total_frames': sum(len(frames) for frames in splits.values()),
        'splits': {
            split: {
                'count': len(frames),
                'percentage': 100 * len(frames) / sum(len(f) for f in splits.values())
            }
            for split, frames in splits.items()
        },
        'seed': splitter.seed
    }

    return summary
