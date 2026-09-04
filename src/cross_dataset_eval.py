"""
Cross-dataset generalization evaluation.
Train on one dataset (FF++), evaluate on another (Celeb-DF).
Measures real-world robustness and generalization gaps.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
import json
from pathlib import Path
import logging

from models_baseline import create_model
from evaluate import DeepfakeEvaluator

logger = logging.getLogger(__name__)


class CrossDatasetEvaluator:
    """
    Evaluate model trained on one dataset on a different dataset.
    Quantifies generalization gaps and distribution shifts.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize evaluator with pretrained model.

        Args:
            model_path: Path to trained model checkpoint
            device: Computation device
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Infer model architecture from checkpoint
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            config = checkpoint['model_config']
            self.model = create_model(
                config.get('architecture', 'efficientnet'),
                num_classes=config.get('num_classes', 2)
            )
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume EfficientNet if no config
            self.model = create_model('efficientnet', num_classes=2)
            if isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model from {self.model_path}")

    def evaluate_on_dataset(
        self,
        test_loader: DataLoader,
        dataset_name: str = "test_dataset"
    ) -> Dict:
        """
        Evaluate model on test dataset.

        Args:
            test_loader: DataLoader for test set
            dataset_name: Name of test dataset for reporting

        Returns:
            Dictionary with evaluation metrics
        """
        evaluator = DeepfakeEvaluator()
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    targets = batch['label'].to(self.device)
                    manipulation_types = batch.get('manipulation_type', None)
                else:
                    images, targets = batch
                    manipulation_types = None

                logits = self.model(images)
                evaluator.add_batch(logits, targets, manipulation_types=manipulation_types)
                total_samples += images.shape[0]

        # Compute metrics
        overall_metrics = evaluator.compute_metrics()
        manipulation_metrics = evaluator.compute_metrics_by_manipulation_type()
        video_metrics = evaluator.compute_video_level_metrics()

        results = {
            'dataset_name': dataset_name,
            'total_samples': total_samples,
            'overall_metrics': overall_metrics,
            'manipulation_metrics': manipulation_metrics,
            'video_metrics': video_metrics
        }

        return results

    def compare_datasets(
        self,
        train_results: Dict,
        test_results: Dict
    ) -> Dict:
        """
        Compare performance across datasets.

        Args:
            train_results: Results on training dataset
            test_results: Results on test dataset

        Returns:
            Comparison with performance gaps
        """
        train_auc = train_results['overall_metrics']['auc_roc']
        test_auc = test_results['overall_metrics']['auc_roc']
        auc_gap = train_auc - test_auc
        auc_gap_pct = (auc_gap / train_auc * 100) if train_auc > 0 else 0

        train_acc = train_results['overall_metrics']['accuracy']
        test_acc = test_results['overall_metrics']['accuracy']
        acc_gap = train_acc - test_acc
        acc_gap_pct = (acc_gap / train_acc * 100) if train_acc > 0 else 0

        comparison = {
            'train_dataset': train_results['dataset_name'],
            'test_dataset': test_results['dataset_name'],
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'auc_gap': float(auc_gap),
            'auc_gap_percentage': float(auc_gap_pct),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'accuracy_gap': float(acc_gap),
            'accuracy_gap_percentage': float(acc_gap_pct),
            'train_eer': float(train_results['overall_metrics']['eer']),
            'test_eer': float(test_results['overall_metrics']['eer']),
            'generalization_score': self._compute_generalization_score(train_auc, test_auc)
        }

        return comparison

    def _compute_generalization_score(self, train_auc: float, test_auc: float) -> float:
        """
        Compute generalization score (0-1).
        1.0 = perfect generalization, 0.0 = no generalization.
        """
        if train_auc < 0.5 or test_auc < 0.5:
            return 0.0

        # Normalize to 0-1 scale
        max_auc = 1.0
        norm_train = (train_auc - 0.5) / (max_auc - 0.5)
        norm_test = (test_auc - 0.5) / (max_auc - 0.5)

        # Generalization = test AUC / train AUC (penalty for drop)
        score = min(1.0, norm_test / norm_train) if norm_train > 0 else 0.0
        return float(score)

    def generate_report(
        self,
        train_results: Dict,
        test_results: Dict,
        output_path: str = "cross_dataset_report.json"
    ) -> str:
        """
        Generate comprehensive cross-dataset report.

        Args:
            train_results: Results on training dataset
            test_results: Results on test dataset
            output_path: Path to save JSON report

        Returns:
            Formatted report string
        """
        comparison = self.compare_datasets(train_results, test_results)

        report = {
            'train_results': train_results,
            'test_results': test_results,
            'comparison': comparison
        }

        # Save JSON report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")

        return self._format_report(comparison, train_results, test_results)

    def _format_report(self, comparison: Dict, train_results: Dict, test_results: Dict) -> str:
        """Format report for display."""
        lines = []
        lines.append("=" * 80)
        lines.append("CROSS-DATASET GENERALIZATION EVALUATION")
        lines.append("=" * 80)

        lines.append(f"\n[DATASETS]")
        lines.append(f"  Training:  {comparison['train_dataset']}")
        lines.append(f"  Testing:   {comparison['test_dataset']}")

        lines.append(f"\n[OVERALL PERFORMANCE]")
        lines.append(f"  Train AUC-ROC:     {comparison['train_auc']:.4f}")
        lines.append(f"  Test AUC-ROC:      {comparison['test_auc']:.4f}")
        lines.append(f"  AUC Gap:           {comparison['auc_gap']:.4f} ({comparison['auc_gap_percentage']:.1f}%)")
        lines.append(f"")
        lines.append(f"  Train Accuracy:    {comparison['train_accuracy']:.4f}")
        lines.append(f"  Test Accuracy:     {comparison['test_accuracy']:.4f}")
        lines.append(f"  Accuracy Gap:      {comparison['accuracy_gap']:.4f} ({comparison['accuracy_gap_percentage']:.1f}%)")
        lines.append(f"")
        lines.append(f"  Train EER:         {comparison['train_eer']:.4f}")
        lines.append(f"  Test EER:          {comparison['test_eer']:.4f}")
        lines.append(f"")
        lines.append(f"  Generalization Score: {comparison['generalization_score']:.3f}")

        # Per-manipulation type
        train_manip = train_results.get('manipulation_metrics', {})
        test_manip = test_results.get('manipulation_metrics', {})

        if train_manip and test_manip:
            lines.append(f"\n[PER-MANIPULATION TYPE GENERALIZATION]")
            for manip_type in sorted(set(train_manip.keys()) | set(test_manip.keys())):
                train_auc_m = train_manip.get(manip_type, {}).get('auc_roc', 0.0)
                test_auc_m = test_manip.get(manip_type, {}).get('auc_roc', 0.0)
                gap_m = train_auc_m - test_auc_m if train_auc_m > 0 else 0

                lines.append(f"\n  {manip_type.upper()}:")
                lines.append(f"    Train AUC: {train_auc_m:.4f}")
                lines.append(f"    Test AUC:  {test_auc_m:.4f}")
                lines.append(f"    Gap:       {gap_m:.4f}")

        lines.append(f"\n[INTERPRETATION]")
        score = comparison['generalization_score']
        if score > 0.9:
            lines.append("  Status: EXCELLENT generalization across datasets")
        elif score > 0.75:
            lines.append("  Status: GOOD generalization, minor distribution shift")
        elif score > 0.5:
            lines.append("  Status: MODERATE generalization, notable performance drop")
        else:
            lines.append("  Status: POOR generalization, significant distribution mismatch")

        lines.append(f"\n  The {comparison['auc_gap_percentage']:.1f}% AUC drop is {'expected' if comparison['auc_gap_percentage'] < 20 else 'notable'}.")
        lines.append(f"  Consider: dataset differences, image quality, camera angles, lighting.")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


def run_cross_dataset_evaluation(
    model_path: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_dataset_name: str = "FaceForensics++",
    test_dataset_name: str = "Celeb-DF",
    output_dir: str = "outputs/cross_dataset",
    device: str = "cuda"
) -> Dict:
    """
    Run full cross-dataset evaluation pipeline.

    Args:
        model_path: Path to trained model
        train_loader: DataLoader for training dataset
        test_loader: DataLoader for test dataset
        train_dataset_name: Name of training dataset
        test_dataset_name: Name of test dataset
        output_dir: Directory to save results
        device: Computation device

    Returns:
        Comprehensive evaluation results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluator = CrossDatasetEvaluator(model_path, device)

    print(f"Evaluating on {train_dataset_name}...")
    train_results = evaluator.evaluate_on_dataset(train_loader, train_dataset_name)

    print(f"Evaluating on {test_dataset_name}...")
    test_results = evaluator.evaluate_on_dataset(test_loader, test_dataset_name)

    # Generate report
    report_path = f"{output_dir}/cross_dataset_report.json"
    report_text = evaluator.generate_report(train_results, test_results, report_path)
    print(report_text)

    # Save text report
    text_report_path = f"{output_dir}/cross_dataset_report.txt"
    with open(text_report_path, 'w') as f:
        f.write(report_text)

    return {
        'train_results': train_results,
        'test_results': test_results,
        'comparison': evaluator.compare_datasets(train_results, test_results),
        'report_path': report_path,
        'text_report_path': text_report_path
    }
