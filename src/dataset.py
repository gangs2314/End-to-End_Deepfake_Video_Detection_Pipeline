import os
from typing import List, Optional, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data_split import DatasetSplitter


class DeepfakeDataset(Dataset):
    """
    Deepfake video detection dataset with video-level splits to prevent leakage.
    Supports frame-based classification with optional video-level metadata.
    """

    def __init__(
        self,
        frame_paths: Optional[List[str]] = None,
        root_dir: Optional[str] = None,
        split: str = "train",
        augment: bool = False,
        manipulation_type: Optional[str] = None
    ):
        """
        Initialize dataset.

        Args:
            frame_paths: List of frame file paths (from DatasetSplitter)
            root_dir: Root directory with frames (for backward compatibility)
            split: "train", "val", or "test"
            augment: Apply data augmentation
            manipulation_type: Filter by specific manipulation type (optional)
        """
        self.images = []
        self.labels = []
        self.manipulation_types = []
        self.split = split

        if frame_paths:
            self._load_from_paths(frame_paths, manipulation_type)
        elif root_dir:
            self._load_from_directory(root_dir)
        else:
            raise ValueError("Either frame_paths or root_dir must be provided")

        self.transform = self._get_transforms(augment)
        print(f"Loaded {len(self.images)} frames for {split} split")

    def _load_from_paths(self, frame_paths: List[str], manipulation_type: Optional[str]):
        """Load frames from explicit paths."""
        for frame_path in frame_paths:
            if not os.path.exists(frame_path):
                continue

            # Determine label and manipulation type
            label = 0 if "real" in frame_path.lower() else 1
            manip_type = self._extract_manipulation_type(frame_path)

            if manipulation_type and manip_type != manipulation_type:
                continue

            self.images.append(frame_path)
            self.labels.append(label)
            self.manipulation_types.append(manip_type)

    def _load_from_directory(self, root_dir: str):
        """Load frames from directory structure (backward compatible)."""
        self.classes = {
            "real": 0,
            "ai_generated": 1,
            "ai_edited": 1
        }

        print(f"Scanning dataset in {root_dir}...")
        for cls_name, label in self.classes.items():
            folder = os.path.join(root_dir, cls_name)
            if os.path.exists(folder):
                count = 0
                for img in os.listdir(folder):
                    if img.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(folder, img))
                        self.labels.append(label)
                        self.manipulation_types.append(cls_name)
                        count += 1
                print(f"  Loaded {count} images from {cls_name}")

    def _extract_manipulation_type(self, frame_path: str) -> str:
        """Extract manipulation type from frame path."""
        path_lower = frame_path.lower()
        if "real" in path_lower:
            return "real"
        elif "deepfake" in path_lower:
            return "deepfakes"
        elif "face2face" in path_lower:
            return "face2face"
        elif "faceswap" in path_lower:
            return "faceswap"
        elif "neuraltexture" in path_lower:
            return "neuraltextures"
        else:
            return "unknown"

    def _get_transforms(self, augment: bool) -> transforms.Compose:
        """Get image transformations."""
        if augment and self.split == "train":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get item.

        Returns:
            Dict with keys: 'image', 'label', 'manipulation_type'
        """
        try:
            image = Image.open(self.images[index]).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {self.images[index]}: {e}")
            # Return black image on error
            image = torch.zeros((3, 224, 224))

        label = torch.tensor(self.labels[index], dtype=torch.long)
        manip_type = self.manipulation_types[index]

        return {
            'image': image,
            'label': label,
            'manipulation_type': manip_type,
            'path': self.images[index]
        }

    def get_split_stats(self) -> Dict:
        """Get statistics about this split."""
        unique_types = set(self.manipulation_types)
        type_counts = {}
        for manip_type in unique_types:
            type_counts[manip_type] = self.manipulation_types.count(manip_type)

        return {
            'total_frames': len(self),
            'real_frames': sum(1 for l in self.labels if l == 0),
            'fake_frames': sum(1 for l in self.labels if l == 1),
            'manipulation_types': type_counts
        }