# Dataset Documentation

## Overview

This project uses video-identity-level splits to prevent data leakage between train/test sets. This is critical for deepfake detection, as frame-level splits would allow the model to memorize unique characteristics of individual videos rather than learning generalizable deepfake detection patterns.

## Splitting Strategy

### Video-Identity Level Splits

Unlike naive frame-level splits, we organize data by **source video** and **actor identity**:

- **Real videos**: All frames from the same real person's video are kept together
- **Fake videos**: All frames from the same fake video (regardless of manipulation type) are kept together

This ensures:
- No frames from the same video appear in both train and test sets
- Model generalizes to unseen identities and videos
- Results reflect real-world deployment performance

### Supported Datasets

1. **FaceForensics++ (FF++)**
   - Raw (c23) with highest quality
   - Includes manipulation types: Deepfakes, Face2Face, FaceSwap, NeuralTextures
   - 1000 videos total (~500 real, ~500 fake)

2. **Celeb-DF**
   - Higher quality deepfakes with more challenging videos
   - ~500 videos total
   - Used for cross-dataset evaluation

3. **DFDC**
   - Diverse manipulation types and camera conditions
   - ~10k videos
   - Used for robustness evaluation

## Current Split Configuration

```yaml
# From configs/default.yaml
dataset:
  split_method: "video_identity"
  test_split: 0.2      # 20% for testing
  val_split: 0.1       # 10% for validation
  seed: 42             # For reproducibility
  frame_sample_strategy: "uniform"
  frames_per_video: 8
```

### Split Ratios

- **Training**: 70% of videos
- **Validation**: 10% of videos
- **Testing**: 20% of videos

Each set is balanced by real/fake ratio of the source dataset.

## Frame Sampling

**Strategy**: Uniform sampling
- Sample N frames evenly distributed across each video
- Default: 8 frames per video
- Prevents temporal bias and reduces storage requirements

**Alternative strategies**:
- Motion-based: Sample frames with highest optical flow (captures manipulation artifacts)
- Random: Random sampling with fixed seed for reproducibility

## Reproducibility

All splits are generated with `seed=42`. To regenerate splits identically:

```python
from src.data_split import DatasetSplitter

splitter = DatasetSplitter(seed=42)
splits = splitter.split_by_video_identity(
    data_root="data/frames",
    test_ratio=0.2,
    val_ratio=0.1
)
splitter.save_split("data/split_config.json")
```

The split configuration (including frame paths) is saved to `data/split_config.json` for consistency across experiment runs.

## Anti-Leakage Guarantees

✓ **No video-level leakage**: Source videos never split between train/test  
✓ **No identity-level leakage**: All frames of same person's videos in same split  
✓ **No frame mixing**: Frames from same video always in same split  
✓ **Reproducible**: Seed ensures identical splits across runs  
✓ **Balanced**: Real/fake ratio maintained across splits

## Usage in Training

```python
from src.data_split import DatasetSplitter

# Generate splits
splitter = DatasetSplitter(seed=42)
splits = splitter.split_by_video_identity("data/frames")

# Use in DataLoader
train_frames = splits['train']
val_frames = splits['val']
test_frames = splits['test']

# Create datasets
train_dataset = DeepfakeDataset(frame_paths=train_frames, split="train")
val_dataset = DeepfakeDataset(frame_paths=val_frames, split="val")
test_dataset = DeepfakeDataset(frame_paths=test_frames, split="test")
```

## Metrics by Manipulation Type

When evaluating on FaceForensics++, results are broken down by manipulation:

| Type | Train Videos | Test Videos | Notes |
|------|-------------|------------|-------|
| Real | ~350 | ~150 | Baseline class |
| Deepfakes | ~140 | ~60 | Most challenging |
| Face2Face | ~140 | ~60 | Expression reenactment |
| FaceSwap | ~140 | ~60 | Identity swap |
| NeuralTextures | ~140 | ~60 | Rendering-based |

## Cross-Dataset Evaluation

For evaluating model robustness:

1. **Train on FF++ → Test on Celeb-DF**
   - Measures generalization to different camera conditions and quality
   - Expected: Performance drop (different distribution)

2. **Train on Celeb-DF → Test on FF++**
   - Reverse direction generalization test
   - Expected: Different performance profile

Results are reported honestly in README, including any significant performance gaps.
