"""
Grad-CAM visualization for deepfake detection model explainability.
Generates heatmaps showing which image regions the model uses for predictions.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List
from pathlib import Path


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.
    Shows which regions of the image contribute to the model's predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Name of layer to compute gradients for
                         (default: automatically find last conv layer)
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        # Find target layer if not specified
        if self.target_layer is None:
            self.target_layer = self._find_last_conv_layer()

        # Get the layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer or module.__class__.__name__ == self.target_layer:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Could not find target layer: {self.target_layer}")

        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = name

        if last_conv is None:
            raise ValueError("Could not find any convolutional layer")

        return last_conv

    def generate_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Target class for CAM (default: predicted class)
            eigen_smooth: Apply eigensmoothing to reduce noise

        Returns:
            CAM heatmap (H, W)
        """
        self.model.eval()

        # Forward pass
        with torch.enable_grad():
            image.requires_grad = True
            logits = self.model(image)

            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Weight activations by gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def overlay_cam_on_image(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            image: Original image (H, W, 3) in RGB format
            cam: CAM heatmap (H, W)
            alpha: Blending factor
            colormap: OpenCV colormap to use

        Returns:
            Overlaid image
        """
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Normalize CAM to 0-255
        cam_normalized = (cam_resized * 255).astype(np.uint8)

        # Apply colormap
        cam_colored = cv2.applyColorMap(cam_normalized, colormap)

        # Convert image to BGR if needed
        if image.shape[2] == 3 and image.dtype in [np.uint8, np.float32, np.float64]:
            if image.dtype in [np.float32, np.float64]:
                image_bgr = (image * 255).astype(np.uint8)
            else:
                image_bgr = image.copy()
            # Assume input is RGB, convert to BGR
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Blend
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, cam_colored, alpha, 0)

        return overlay

    def process_batch(
        self,
        images: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Process batch of images and generate CAMs.

        Args:
            images: Batch of images (B, 3, H, W)
            target_classes: Target class for each image

        Returns:
            List of CAM heatmaps
        """
        cams = []
        for i in range(images.shape[0]):
            image_i = images[i:i+1]
            target = target_classes[i] if target_classes else None
            cam = self.generate_cam(image_i, target)
            cams.append(cam)

        return cams


class ExplainabilityPipeline:
    """
    Complete pipeline for generating and visualizing model explanations.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """
        Initialize pipeline.

        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM
        """
        self.gradcam = GradCAM(model, target_layer)
        self.model = model

    def explain_prediction(
        self,
        image: torch.Tensor,
        image_original: Optional[np.ndarray] = None,
        target_class: Optional[int] = None
    ) -> dict:
        """
        Generate complete explanation for a single prediction.

        Args:
            image: Input tensor (1, 3, H, W)
            image_original: Original image for overlay (H, W, 3)
            target_class: Target class (optional)

        Returns:
            Dictionary with predictions, confidence, and heatmap
        """
        self.model.eval()

        # Get prediction
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        # Generate Grad-CAM
        cam = self.gradcam.generate_cam(image.clone(), target_class=target_class)

        # Create overlay if original image provided
        overlay = None
        if image_original is not None:
            overlay = self.gradcam.overlay_cam_on_image(image_original, cam)

        return {
            'prediction': pred_class,
            'confidence': float(confidence),
            'probabilities': probs[0].cpu().numpy(),
            'cam': cam,
            'overlay': overlay,
            'class_names': ['Real', 'Fake']
        }

    def explain_batch(
        self,
        images: torch.Tensor,
        images_original: Optional[List[np.ndarray]] = None,
        target_classes: Optional[List[int]] = None
    ) -> List[dict]:
        """
        Generate explanations for batch of images.

        Args:
            images: Batch of tensors (B, 3, H, W)
            images_original: List of original images for overlays
            target_classes: List of target classes

        Returns:
            List of explanation dictionaries
        """
        explanations = []

        for i in range(images.shape[0]):
            image_i = images[i:i+1]
            image_orig_i = images_original[i] if images_original else None
            target_i = target_classes[i] if target_classes else None

            exp = self.explain_prediction(image_i, image_orig_i, target_i)
            explanations.append(exp)

        return explanations

    def save_visualization(
        self,
        explanation: dict,
        output_dir: str,
        name: str,
        save_cam: bool = True,
        save_overlay: bool = True
    ):
        """
        Save visualization outputs.

        Args:
            explanation: Explanation dictionary from explain_prediction
            output_dir: Directory to save images
            name: Base name for files
            save_cam: Save CAM heatmap
            save_overlay: Save overlay on original image
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if save_cam:
            cam_image = (explanation['cam'] * 255).astype(np.uint8)
            cam_colored = cv2.applyColorMap(cam_image, cv2.COLORMAP_JET)
            cv2.imwrite(
                str(Path(output_dir) / f"{name}_cam.jpg"),
                cam_colored
            )

        if save_overlay and explanation['overlay'] is not None:
            cv2.imwrite(
                str(Path(output_dir) / f"{name}_overlay.jpg"),
                explanation['overlay']
            )

        # Save metadata
        metadata = {
            'prediction': 'FAKE' if explanation['prediction'] == 1 else 'REAL',
            'confidence': float(explanation['confidence']),
            'probabilities': {
                'real': float(explanation['probabilities'][0]),
                'fake': float(explanation['probabilities'][1])
            }
        }

        import json
        with open(Path(output_dir) / f"{name}_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def visualize_attention_regions(
    image: np.ndarray,
    cam: np.ndarray,
    threshold: float = 0.5,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization highlighting attention regions.

    Args:
        image: Original image (H, W, 3)
        cam: CAM heatmap (H, W)
        threshold: Threshold for highlighting
        output_path: Optional path to save

    Returns:
        Visualization image
    """
    # Resize CAM
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

    # Create mask
    mask = (cam_resized > threshold).astype(np.uint8) * 255

    # Create visualization
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    visualization = image_bgr.copy()
    cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, visualization)

    return visualization
