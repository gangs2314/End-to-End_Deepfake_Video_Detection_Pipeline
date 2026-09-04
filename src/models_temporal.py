"""
Temporal models for deepfake detection.
Capture inter-frame inconsistencies by modeling sequences of frames.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from models_baseline import BaselineFrameClassifier


class TemporalLSTMClassifier(nn.Module):
    """
    LSTM-based temporal model.
    Processes sequence of frame-level features to detect temporal inconsistencies.
    """

    def __init__(
        self,
        frame_model_name: str = "efficientnet",
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        num_classes: int = 2
    ):
        """
        Initialize temporal LSTM model.

        Args:
            frame_model_name: Baseline frame classifier ("efficientnet", "xception", "resnet50")
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            num_classes: Output classes
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Freeze baseline frame classifier and extract features
        self.frame_model = BaselineFrameClassifier(frame_model_name, num_classes=2)
        self.frame_model.eval()
        for param in self.frame_model.parameters():
            param.requires_grad = False

        # Get feature dimension from baseline model
        self.feature_dim = self._get_feature_dim()

        # Temporal encoding with LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Classification head
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _get_feature_dim(self) -> int:
        """Get feature dimension from frame model."""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.frame_model.get_backbone_output(dummy_input)
        return features.shape[1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_frames, 3, 224, 224)

        Returns:
            Tuple of (logits, frame_features) where logits shape is (batch_size, num_classes)
        """
        batch_size, num_frames, c, h, w = x.shape

        # Extract features for each frame
        x_reshaped = x.view(batch_size * num_frames, c, h, w)
        with torch.no_grad():
            frame_features = self.frame_model.get_backbone_output(x_reshaped)

        # Reshape for LSTM
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(frame_features)

        # Use last output for classification
        last_output = lstm_out[:, -1, :]

        # Classification
        logits = self.classifier(last_output)

        return logits, frame_features

    def get_frame_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get per-frame predictions from baseline model."""
        batch_size, num_frames, c, h, w = x.shape
        x_reshaped = x.view(batch_size * num_frames, c, h, w)

        with torch.no_grad():
            frame_logits = self.frame_model(x_reshaped)
            frame_logits = frame_logits.view(batch_size, num_frames, -1)

        return frame_logits


class TemporalGRUClassifier(nn.Module):
    """
    GRU-based temporal model (lighter than LSTM).
    """

    def __init__(
        self,
        frame_model_name: str = "efficientnet",
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        num_classes: int = 2
    ):
        """
        Initialize temporal GRU model.

        Args:
            frame_model_name: Baseline frame classifier
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Use bidirectional GRU
            num_classes: Output classes
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Freeze baseline frame classifier
        self.frame_model = BaselineFrameClassifier(frame_model_name, num_classes=2)
        self.frame_model.eval()
        for param in self.frame_model.parameters():
            param.requires_grad = False

        # Get feature dimension
        self.feature_dim = self._get_feature_dim()

        # Temporal encoding with GRU
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Classification head
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _get_feature_dim(self) -> int:
        """Get feature dimension from frame model."""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.frame_model.get_backbone_output(dummy_input)
        return features.shape[1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size, num_frames, c, h, w = x.shape

        # Extract features for each frame
        x_reshaped = x.view(batch_size * num_frames, c, h, w)
        with torch.no_grad():
            frame_features = self.frame_model.get_backbone_output(x_reshaped)

        # Reshape for GRU
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # GRU encoding
        gru_out, h_n = self.gru(frame_features)

        # Use last output
        last_output = gru_out[:, -1, :]

        # Classification
        logits = self.classifier(last_output)

        return logits, frame_features


class TemporalAttentionClassifier(nn.Module):
    """
    Attention-based temporal model.
    Uses self-attention to weight important frames in the sequence.
    """

    def __init__(
        self,
        frame_model_name: str = "efficientnet",
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 2
    ):
        """
        Initialize attention-based model.

        Args:
            frame_model_name: Baseline frame classifier
            hidden_dim: Transformer hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            num_classes: Output classes
        """
        super().__init__()
        self.num_classes = num_classes

        # Freeze baseline frame classifier
        self.frame_model = BaselineFrameClassifier(frame_model_name, num_classes=2)
        self.frame_model.eval()
        for param in self.frame_model.parameters():
            param.requires_grad = False

        # Get feature dimension
        self.feature_dim = self._get_feature_dim()

        # Project to hidden dimension
        self.input_projection = nn.Linear(self.feature_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _get_feature_dim(self) -> int:
        """Get feature dimension from frame model."""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.frame_model.get_backbone_output(dummy_input)
        return features.shape[1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size, num_frames, c, h, w = x.shape

        # Extract features for each frame
        x_reshaped = x.view(batch_size * num_frames, c, h, w)
        with torch.no_grad():
            frame_features = self.frame_model.get_backbone_output(x_reshaped)

        # Reshape and project
        frame_features = frame_features.view(batch_size, num_frames, -1)
        projected = self.input_projection(frame_features)

        # Transformer encoding
        attention_out = self.transformer(projected)

        # Use mean pooling over sequence
        pooled = attention_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits, frame_features


def create_temporal_model(
    model_type: str = "lstm",
    frame_model_name: str = "efficientnet",
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    num_classes: int = 2,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Factory function to create temporal models.

    Args:
        model_type: "lstm", "gru", or "attention"
        frame_model_name: Baseline model architecture
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        num_classes: Output classes
        checkpoint_path: Path to pretrained weights

    Returns:
        Model instance
    """
    if model_type == "lstm":
        model = TemporalLSTMClassifier(
            frame_model_name, hidden_dim, num_layers, dropout, num_classes=num_classes
        )
    elif model_type == "gru":
        model = TemporalGRUClassifier(
            frame_model_name, hidden_dim, num_layers, dropout, num_classes=num_classes
        )
    elif model_type == "attention":
        model = TemporalAttentionClassifier(
            frame_model_name, hidden_dim, num_layers=num_layers,
            dropout=dropout, num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model
