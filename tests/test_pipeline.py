"""
Pytest tests for deepfake detection pipeline.
Tests for preprocessing, models, and evaluation.
"""
import pytest
import torch
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import DeepfakeDataset
from models_baseline import create_model, get_model_info
from models_temporal import create_temporal_model
from evaluate import DeepfakeEvaluator
from data_split import DatasetSplitter


class TestDatasetSplitter:
    """Tests for video-level dataset splitting."""

    def test_splitter_initialization(self):
        """Test DatasetSplitter initialization."""
        splitter = DatasetSplitter(seed=42)
        assert splitter.seed == 42

    def test_video_grouping(self):
        """Test frame grouping by video."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake frame structure
            real_dir = Path(tmpdir) / "real"
            real_dir.mkdir()

            # Create dummy frames
            for i in range(3):
                (real_dir / f"video1.mp4_{i}.jpg").touch()
            for i in range(2):
                (real_dir / f"video2.mp4_{i}.jpg").touch()

            splitter = DatasetSplitter()
            video_groups = splitter._group_frames_by_video(str(real_dir))

            assert len(video_groups) == 2
            assert "video1.mp4" in video_groups
            assert "video2.mp4" in video_groups
            assert len(video_groups["video1.mp4"]) == 3
            assert len(video_groups["video2.mp4"]) == 2

    def test_split_proportions(self):
        """Test that splits maintain correct proportions."""
        splitter = DatasetSplitter(seed=42)

        videos = [
            (f"video_{i}.mp4", [f"frame_{j}.jpg" for j in range(10)])
            for i in range(10)
        ]

        split = splitter._split_video_list(videos, test_ratio=0.2, val_ratio=0.1)

        total = len(split['train']) + len(split['val']) + len(split['test'])
        train_ratio = len(split['train']) / total
        val_ratio = len(split['val']) / total
        test_ratio = len(split['test']) / total

        # Check approximate proportions (70/10/20)
        assert 0.6 < train_ratio < 0.8
        assert 0.05 < val_ratio < 0.15
        assert 0.15 < test_ratio < 0.25

    def test_reproducibility(self):
        """Test that same seed produces same splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / "real"
            real_dir.mkdir()

            for i in range(5):
                for j in range(3):
                    (real_dir / f"video_{i}.mp4_{j}.jpg").touch()

            # First split
            splitter1 = DatasetSplitter(seed=42)
            splits1 = splitter1.split_by_video_identity(str(real_dir))

            # Second split with same seed
            splitter2 = DatasetSplitter(seed=42)
            splits2 = splitter2.split_by_video_identity(str(real_dir))

            # Should be identical
            assert len(splits1['train']) == len(splits2['train'])
            assert len(splits1['val']) == len(splits2['val'])
            assert len(splits1['test']) == len(splits2['test'])


class TestDeepfakeDataset:
    """Tests for DeepfakeDataset."""

    def test_dataset_creation_from_paths(self):
        """Test dataset creation from frame paths."""
        frame_paths = [f"frame_{i}.jpg" for i in range(10)]
        # Mock paths - in real scenario these would be actual files
        dataset = DeepfakeDataset(frame_paths=frame_paths, split="train")
        assert len(dataset) == 10

    def test_manipulation_type_extraction(self):
        """Test extraction of manipulation type from paths."""
        dataset = DeepfakeDataset(frame_paths=["dummy.jpg"], split="train")

        assert dataset._extract_manipulation_type("deepfake_frame.jpg") == "deepfakes"
        assert dataset._extract_manipulation_type("face2face_frame.jpg") == "face2face"
        assert dataset._extract_manipulation_type("faceswap_frame.jpg") == "faceswap"
        assert dataset._extract_manipulation_type("real_frame.jpg") == "real"

    def test_label_from_manipulation_type(self):
        """Test that manipulation type produces correct label."""
        dataset = DeepfakeDataset(frame_paths=["dummy.jpg"], split="train")

        assert dataset._extract_manipulation_type("real_frame.jpg") == "real"
        # Anything not "real" is fake
        assert dataset._extract_manipulation_type("deepfake_frame.jpg") != "real"


class TestBaselineModels:
    """Tests for baseline frame classifiers."""

    def test_efficientnet_creation(self):
        """Test EfficientNet model creation."""
        model = create_model("efficientnet", num_classes=2)
        assert model is not None
        assert model.num_classes == 2

    def test_resnet50_creation(self):
        """Test ResNet50 model creation."""
        model = create_model("resnet50", num_classes=2)
        assert model is not None
        assert model.num_classes == 2

    def test_model_forward_pass(self):
        """Test forward pass through model."""
        model = create_model("resnet50", num_classes=2)
        model.eval()

        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2)  # batch_size=2, num_classes=2

    def test_model_feature_extraction(self):
        """Test feature extraction from model."""
        model = create_model("efficientnet", num_classes=2)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.get_backbone_output(x)

        # Should be 2D (batch, features)
        assert len(features.shape) == 2
        assert features.shape[0] == 1

    def test_model_info(self):
        """Test model information retrieval."""
        model = create_model("efficientnet", num_classes=2)
        info = get_model_info(model)

        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['frozen_parameters'] >= 0
        assert info['num_classes'] == 2

    def test_freeze_backbone(self):
        """Test backbone freezing."""
        model = create_model("efficientnet", num_classes=2, freeze_backbone=True)
        info = get_model_info(model)

        # Backbone should be frozen
        assert info['frozen_parameters'] > 0

    def test_gradient_flow(self):
        """Test that gradients flow through model."""
        model = create_model("resnet50", num_classes=2)
        model.train()

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTemporalModels:
    """Tests for temporal modeling."""

    def test_lstm_model_creation(self):
        """Test LSTM model creation."""
        model = create_temporal_model("lstm", num_classes=2)
        assert model is not None

    def test_gru_model_creation(self):
        """Test GRU model creation."""
        model = create_temporal_model("gru", num_classes=2)
        assert model is not None

    def test_attention_model_creation(self):
        """Test attention model creation."""
        model = create_temporal_model("attention", num_classes=2)
        assert model is not None

    def test_temporal_forward_pass(self):
        """Test forward pass through temporal model."""
        model = create_temporal_model("lstm", num_classes=2)
        model.eval()

        # Create dummy sequence (batch, num_frames, channels, height, width)
        x = torch.randn(2, 8, 3, 224, 224)
        with torch.no_grad():
            logits, features = model(x)

        assert logits.shape == (2, 2)  # batch_size=2, num_classes=2
        assert features.shape[0] == 2

    def test_frame_level_predictions(self):
        """Test frame-level predictions from temporal model."""
        model = create_temporal_model("lstm", num_classes=2)
        model.eval()

        x = torch.randn(1, 4, 3, 224, 224)
        with torch.no_grad():
            frame_preds = model.get_frame_predictions(x)

        assert frame_preds.shape[0] == 1  # batch_size
        assert frame_preds.shape[1] == 4  # num_frames
        assert frame_preds.shape[2] == 2  # num_classes


class TestEvaluator:
    """Tests for evaluation metrics."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = DeepfakeEvaluator()
        assert len(evaluator.targets) == 0

    def test_add_batch(self):
        """Test adding batch to evaluator."""
        evaluator = DeepfakeEvaluator()

        logits = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        targets = torch.tensor([0, 1, 0])

        evaluator.add_batch(logits, targets)

        assert len(evaluator.targets) == 3
        assert len(evaluator.predictions) == 3
        assert len(evaluator.probabilities) == 3

    def test_metrics_computation(self):
        """Test metrics computation."""
        evaluator = DeepfakeEvaluator()

        # Create perfect predictions
        logits = torch.tensor([
            [2.0, -2.0],  # Real (class 0)
            [-2.0, 2.0],  # Fake (class 1)
            [2.0, -2.0],  # Real
            [-2.0, 2.0],  # Fake
        ])
        targets = torch.tensor([0, 1, 0, 1])

        evaluator.add_batch(logits, targets)
        metrics = evaluator.compute_metrics()

        assert 'accuracy' in metrics
        assert 'auc_roc' in metrics
        assert 'eer' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect predictions

    def test_eer_computation(self):
        """Test Equal Error Rate computation."""
        evaluator = DeepfakeEvaluator()

        targets = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.3, 0.7, 0.9])

        eer = evaluator._compute_eer(targets, probabilities)

        assert 0.0 <= eer <= 1.0

    def test_video_level_aggregation(self):
        """Test video-level metric aggregation."""
        evaluator = DeepfakeEvaluator()

        logits = torch.tensor([
            [2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        targets = torch.tensor([0, 1, 0, 1])
        video_ids = ["video1", "video1", "video2", "video2"]

        evaluator.add_batch(logits, targets, video_ids=video_ids)
        video_metrics = evaluator.compute_video_level_metrics()

        assert video_metrics['num_videos'] == 2
        assert 'accuracy' in video_metrics
        assert 'auc_roc' in video_metrics

    def test_per_manipulation_type_metrics(self):
        """Test per-manipulation type breakdown."""
        evaluator = DeepfakeEvaluator()

        logits = torch.tensor([
            [2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        targets = torch.tensor([0, 1, 0, 1])
        manipulation_types = ["real", "deepfakes", "real", "deepfakes"]

        evaluator.add_batch(
            logits, targets, manipulation_types=manipulation_types
        )
        metrics = evaluator.compute_metrics_by_manipulation_type()

        assert "real" in metrics
        assert "deepfakes" in metrics
        assert metrics["real"]['count'] == 2
        assert metrics["deepfakes"]['count'] == 2

    def test_report_generation(self):
        """Test report generation."""
        evaluator = DeepfakeEvaluator()

        logits = torch.tensor([
            [2.0, -2.0],
            [-2.0, 2.0],
        ])
        targets = torch.tensor([0, 1])

        evaluator.add_batch(logits, targets)
        report = evaluator.generate_report()

        assert "EVALUATION REPORT" in report
        assert "Accuracy" in report
        assert "AUC-ROC" in report


class TestIntegration:
    """Integration tests for pipeline."""

    def test_end_to_end_inference(self):
        """Test end-to-end inference pipeline."""
        # Create models
        frame_model = create_model("resnet50", num_classes=2)
        temporal_model = create_temporal_model("lstm", num_classes=2)

        # Create dummy data
        x = torch.randn(1, 8, 3, 224, 224)

        # Frame-level inference
        frame_model.eval()
        with torch.no_grad():
            frame_logits = frame_model(x[:, 0])
            assert frame_logits.shape == (1, 2)

        # Temporal inference
        temporal_model.eval()
        with torch.no_grad():
            temporal_logits, _ = temporal_model(x)
            assert temporal_logits.shape == (1, 2)

    def test_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Generate predictions
        evaluator = DeepfakeEvaluator()

        for _ in range(100):
            logits = torch.randn(8, 2)
            targets = torch.randint(0, 2, (8,))
            evaluator.add_batch(logits, targets)

        # Compute metrics
        metrics = evaluator.compute_metrics()
        assert 'accuracy' in metrics
        assert 'auc_roc' in metrics

        # Generate report
        report = evaluator.generate_report()
        assert len(report) > 0


# Pytest fixtures

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_model():
    """Create dummy model for testing."""
    return create_model("resnet50", num_classes=2)


@pytest.fixture
def dummy_temporal_model():
    """Create dummy temporal model for testing."""
    return create_temporal_model("lstm", num_classes=2)


# Command to run tests
# pytest tests/test_pipeline.py -v --cov=src --cov-report=html
