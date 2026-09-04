"""
Baseline frame classification models: EfficientNet and Xception.
These models process individual frames to detect deepfakes.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Tuple


class BaselineFrameClassifier(nn.Module):
    """
    Base frame-level deepfake classifier.
    Outputs per-frame predictions (0=real, 1=fake).
    """

    def __init__(self, model_name: str = "efficientnet", num_classes: int = 2):
        """
        Initialize baseline classifier.

        Args:
            model_name: "efficientnet", "xception", or "resnet50"
            num_classes: Number of output classes (binary: 2)
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if model_name == "efficientnet":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
        elif model_name == "xception":
            # Xception from timm if available, else use ResNet50 as fallback
            try:
                import timm
                self.backbone = timm.create_model('xception', pretrained=True)
                in_features = self.backbone.fc.in_features
            except ImportError:
                print("timm not available, using ResNet50 instead")
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                in_features = self.backbone.fc.in_features
        else:
            # Default to ResNet50
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self._get_features(x)
        logits = self.classifier(features)
        return logits

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        if self.model_name == "efficientnet":
            features = self.backbone.features(x)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
        elif self.model_name == "xception":
            features = self.backbone.forward_features(x)
        else:
            # ResNet50
            features = self.backbone.layer4(
                self.backbone.layer3(
                    self.backbone.layer2(
                        self.backbone.layer1(
                            self.backbone.conv1(x)
                        )
                    )
                )
            )
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)

        return features

    def get_backbone_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone features (useful for visualization and temporal models)."""
        return self._get_features(x)


class EfficientNetClassifier(BaselineFrameClassifier):
    """EfficientNet-B0 based classifier (lightweight, efficient)."""

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False):
        super().__init__(model_name="efficientnet", num_classes=num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


class XceptionClassifier(BaselineFrameClassifier):
    """Xception based classifier (high accuracy, more parameters)."""

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False):
        super().__init__(model_name="xception", num_classes=num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


class ResNet50Classifier(BaselineFrameClassifier):
    """ResNet50 based classifier (balanced, widely used)."""

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False):
        super().__init__(model_name="resnet50", num_classes=num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


def create_model(
    model_name: str = "efficientnet",
    num_classes: int = 2,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        model_name: "efficientnet", "xception", or "resnet50"
        num_classes: Number of output classes
        freeze_backbone: Freeze backbone weights
        checkpoint_path: Path to pretrained weights

    Returns:
        Model instance
    """
    if model_name == "efficientnet":
        model = EfficientNetClassifier(num_classes, freeze_backbone)
    elif model_name == "xception":
        model = XceptionClassifier(num_classes, freeze_backbone)
    else:
        model = ResNet50Classifier(num_classes, freeze_backbone)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


def get_model_info(model: nn.Module) -> Dict:
    """Get model information (parameters, FLOPs, etc.)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'model_name': getattr(model, 'model_name', 'unknown'),
        'num_classes': getattr(model, 'num_classes', 2)
    }
