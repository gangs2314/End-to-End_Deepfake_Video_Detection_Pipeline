"""
Comprehensive evaluation metrics for deepfake detection.
Includes AUC-ROC, EER, accuracy, and per-manipulation-type breakdown.
"""
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class DeepfakeEvaluator:
    """
    Comprehensive evaluation for deepfake detection models.
    Computes standard metrics and provides detailed analysis.
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize evaluator.

        Args:
            num_classes: Number of output classes (binary: 2)
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.probabilities = []
        self.targets = []
        self.manipulation_types = []
        self.video_ids = []

    def add_batch(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        manipulation_types: Optional[List[str]] = None,
        video_ids: Optional[List[str]] = None
    ):
        """
        Add batch of predictions.

        Args:
            logits: Model output logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            manipulation_types: Manipulation type for each sample
            video_ids: Video ID for each sample
        """
        # Convert to numpy
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Get predictions and probabilities
        probs = self._softmax(logits)
        preds = np.argmax(logits, axis=1)

        self.predictions.extend(preds)
        self.probabilities.extend(probs[:, 1])  # Probability of fake class
        self.targets.extend(targets)

        if manipulation_types:
            self.manipulation_types.extend(manipulation_types)
        else:
            self.manipulation_types.extend(["unknown"] * len(targets))

        if video_ids:
            self.video_ids.extend(video_ids)
        else:
            self.video_ids.extend([f"video_{i}" for i in range(len(targets))])

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def compute_metrics(self) -> Dict:
        """Compute all evaluation metrics."""
        if len(self.targets) == 0:
            return {}

        predictions = np.array(self.predictions)
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)

        # Overall metrics
        metrics = {
            'accuracy': float(accuracy_score(targets, predictions)),
            'auc_roc': float(roc_auc_score(targets, probabilities)),
            'eer': float(self._compute_eer(targets, probabilities)),
            'confusion_matrix': confusion_matrix(targets, predictions).tolist()
        }

        # Per-class metrics
        report = classification_report(targets, predictions, output_dict=True, zero_division=0)
        metrics['precision_real'] = float(report['0']['precision'])
        metrics['recall_real'] = float(report['0']['recall'])
        metrics['f1_real'] = float(report['0']['f1-score'])
        metrics['precision_fake'] = float(report['1']['precision'])
        metrics['recall_fake'] = float(report['1']['recall'])
        metrics['f1_fake'] = float(report['1']['f1-score'])

        return metrics

    def compute_metrics_by_manipulation_type(self) -> Dict:
        """Compute metrics broken down by manipulation type."""
        predictions = np.array(self.predictions)
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)
        manipulation_types = np.array(self.manipulation_types)

        results = {}

        for manip_type in np.unique(manipulation_types):
            mask = manipulation_types == manip_type
            if np.sum(mask) == 0:
                continue

            type_targets = targets[mask]
            type_probs = probabilities[mask]
            type_preds = predictions[mask]

            results[manip_type] = {
                'count': int(np.sum(mask)),
                'accuracy': float(accuracy_score(type_targets, type_preds)),
                'auc_roc': float(roc_auc_score(type_targets, type_probs)) if len(np.unique(type_targets)) > 1 else 0.0,
                'eer': float(self._compute_eer(type_targets, type_probs)),
                'real_count': int(np.sum(type_targets == 0)),
                'fake_count': int(np.sum(type_targets == 1))
            }

            # Per-class metrics
            report = classification_report(type_targets, type_preds, output_dict=True, zero_division=0)
            results[manip_type]['precision_real'] = float(report['0']['precision'])
            results[manip_type]['recall_real'] = float(report['0']['recall'])
            results[manip_type]['f1_real'] = float(report['0']['f1-score'])
            results[manip_type]['precision_fake'] = float(report['1']['precision'])
            results[manip_type]['recall_fake'] = float(report['1']['recall'])
            results[manip_type]['f1_fake'] = float(report['1']['f1-score'])

        return results

    def compute_video_level_metrics(self) -> Dict:
        """
        Compute video-level metrics by aggregating frame-level predictions.
        Each video gets a single verdict based on majority voting or averaging.
        """
        predictions = np.array(self.predictions)
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)
        video_ids = np.array(self.video_ids)

        video_predictions = {}
        video_targets = {}
        video_probs = {}

        # Aggregate by video
        for video_id in np.unique(video_ids):
            mask = video_ids == video_id
            video_targets[video_id] = targets[mask][0]  # All frames should have same target

            # Average probability across frames
            avg_prob = np.mean(probabilities[mask])
            video_probs[video_id] = avg_prob
            video_predictions[video_id] = 1 if avg_prob > 0.5 else 0

        # Convert to arrays
        video_preds_array = np.array([video_predictions[v] for v in sorted(video_predictions.keys())])
        video_targets_array = np.array([video_targets[v] for v in sorted(video_targets.keys())])
        video_probs_array = np.array([video_probs[v] for v in sorted(video_probs.keys())])

        metrics = {
            'num_videos': len(video_predictions),
            'accuracy': float(accuracy_score(video_targets_array, video_preds_array)),
            'auc_roc': float(roc_auc_score(video_targets_array, video_probs_array)),
            'eer': float(self._compute_eer(video_targets_array, video_probs_array))
        }

        return metrics

    def _compute_eer(self, targets: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Compute Equal Error Rate (EER).
        The threshold where FPR = FNR.
        """
        fpr, fnr, thresholds = self._compute_roc_metrics(targets, probabilities)

        # Find threshold where FPR == FNR
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]

        return eer

    def _compute_roc_metrics(self, targets: np.ndarray, probabilities: np.ndarray) -> Tuple:
        """Compute FPR, FNR, and thresholds for ROC curve."""
        fpr, tpr, thresholds = roc_curve(targets, probabilities)
        fnr = 1 - tpr
        return fpr, fnr, thresholds

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            output_path: Optional path to save report as JSON

        Returns:
            Formatted report string
        """
        overall_metrics = self.compute_metrics()
        manipulation_metrics = self.compute_metrics_by_manipulation_type()
        video_metrics = self.compute_video_level_metrics()

        report = {
            'overall': overall_metrics,
            'by_manipulation_type': manipulation_metrics,
            'video_level': video_metrics,
            'total_samples': len(self.targets),
            'real_samples': sum(1 for t in self.targets if t == 0),
            'fake_samples': sum(1 for t in self.targets if t == 1)
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_path}")

        return self._format_report(report)

    def _format_report(self, report: Dict) -> str:
        """Format report for display."""
        lines = []
        lines.append("=" * 80)
        lines.append("DEEPFAKE DETECTION EVALUATION REPORT")
        lines.append("=" * 80)

        # Overall metrics
        lines.append("\n[OVERALL METRICS]")
        overall = report['overall']
        lines.append(f"  Accuracy:      {overall['accuracy']:.4f}")
        lines.append(f"  AUC-ROC:       {overall['auc_roc']:.4f}")
        lines.append(f"  EER:           {overall['eer']:.4f}")
        lines.append(f"  Real Precision: {overall['precision_real']:.4f}")
        lines.append(f"  Real Recall:    {overall['recall_real']:.4f}")
        lines.append(f"  Real F1-Score:  {overall['f1_real']:.4f}")
        lines.append(f"  Fake Precision: {overall['precision_fake']:.4f}")
        lines.append(f"  Fake Recall:    {overall['recall_fake']:.4f}")
        lines.append(f"  Fake F1-Score:  {overall['f1_fake']:.4f}")

        # Sample statistics
        lines.append(f"\n[SAMPLE STATISTICS]")
        lines.append(f"  Total Samples: {report['total_samples']}")
        lines.append(f"  Real Samples:  {report['real_samples']}")
        lines.append(f"  Fake Samples:  {report['fake_samples']}")

        # Per-manipulation type
        if report['by_manipulation_type']:
            lines.append(f"\n[PER-MANIPULATION TYPE METRICS]")
            for manip_type, metrics in sorted(report['by_manipulation_type'].items()):
                lines.append(f"\n  {manip_type.upper()}:")
                lines.append(f"    Count:       {metrics['count']}")
                lines.append(f"    Accuracy:    {metrics['accuracy']:.4f}")
                lines.append(f"    AUC-ROC:     {metrics['auc_roc']:.4f}")
                lines.append(f"    EER:         {metrics['eer']:.4f}")
                lines.append(f"    Precision:   {metrics['precision_fake']:.4f}")
                lines.append(f"    Recall:      {metrics['recall_fake']:.4f}")

        # Video-level metrics
        if report['video_level']['num_videos'] > 0:
            lines.append(f"\n[VIDEO-LEVEL METRICS]")
            video = report['video_level']
            lines.append(f"  Num Videos:  {video['num_videos']}")
            lines.append(f"  Accuracy:    {video['accuracy']:.4f}")
            lines.append(f"  AUC-ROC:     {video['auc_roc']:.4f}")
            lines.append(f"  EER:         {video['eer']:.4f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def plot_roc_curve(self, output_path: Optional[str] = None):
        """
        Plot ROC curve.

        Args:
            output_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping ROC plot")
            return

        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)

        fpr, tpr, _ = roc_curve(targets, probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {output_path}")

        plt.close()

    def plot_confusion_matrix(self, output_path: Optional[str] = None):
        """
        Plot confusion matrix.

        Args:
            output_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping confusion matrix plot")
            return

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        cm = confusion_matrix(targets, predictions)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Real', 'Fake'])
        plt.yticks(tick_marks, ['Real', 'Fake'])

        # Annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")

        plt.close()
