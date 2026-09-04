"""
Experiment tracking integration with Weights & Biases and MLflow.
Logs training metrics, model architecture, hyperparameters, and evaluation results.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified interface for experiment tracking (W&B and MLflow).
    Simplifies logging and comparison across multiple runs.
    """

    def __init__(
        self,
        project_name: str = "deepfake-detection",
        run_name: str = "experiment",
        backend: str = "wandb",
        offline: bool = False
    ):
        """
        Initialize experiment tracker.

        Args:
            project_name: Project name for tracking
            run_name: Name for this run
            backend: "wandb", "mlflow", or "both"
            offline: Run offline (no cloud sync)
        """
        self.project_name = project_name
        self.run_name = run_name
        self.backend = backend
        self.offline = offline
        self.wandb_run = None
        self.mlflow_run = None

        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize tracking backends."""
        if self.backend in ["wandb", "both"]:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.run_name,
                    offline=self.offline
                )
                logger.info("Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"Could not initialize W&B: {e}")

        if self.backend in ["mlflow", "both"]:
            try:
                import mlflow
                mlflow.set_experiment(self.project_name)
                mlflow.start_run(run_name=self.run_name)
                self.mlflow_run = mlflow.active_run()
                logger.info("MLflow initialized")
            except Exception as e:
                logger.warning(f"Could not initialize MLflow: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values
        """
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)

        if self.wandb_run:
            try:
                import wandb
                wandb.config.update(flat_params)
            except Exception as e:
                logger.debug(f"W&B param logging failed: {e}")

        if self.mlflow_run:
            try:
                import mlflow
                for key, value in flat_params.items():
                    mlflow.log_param(key, value)
            except Exception as e:
                logger.debug(f"MLflow param logging failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Step/epoch number
        """
        if self.wandb_run:
            try:
                import wandb
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception as e:
                logger.debug(f"W&B metric logging failed: {e}")

        if self.mlflow_run:
            try:
                import mlflow
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step or 0)
            except Exception as e:
                logger.debug(f"MLflow metric logging failed: {e}")

    def log_model(
        self,
        model: nn.Module,
        model_name: str = "model",
        model_type: str = "pytorch"
    ):
        """
        Log model architecture and parameters.

        Args:
            model: PyTorch model
            model_name: Name for the model
            model_type: Type of model ("pytorch", "onnx", etc.)
        """
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = {
            'model_name': model_name,
            'model_type': model_type,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_class': model.__class__.__name__
        }

        self.log_params({'model': model_info})

        # Log model architecture as text
        arch_str = str(model)
        if self.wandb_run:
            try:
                import wandb
                wandb.log({"model_architecture": wandb.Html(f"<pre>{arch_str}</pre>")})
            except Exception as e:
                logger.debug(f"W&B model logging failed: {e}")

    def log_artifacts(self, artifact_path: str, artifact_type: str = "file"):
        """
        Log file artifacts (checkpoints, plots, reports).

        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact ("model", "plot", "report", etc.)
        """
        path = Path(artifact_path)

        if not path.exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return

        if self.wandb_run:
            try:
                import wandb
                artifact = wandb.Artifact(name=path.stem, type=artifact_type)
                artifact.add_file(artifact_path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.debug(f"W&B artifact logging failed: {e}")

        if self.mlflow_run:
            try:
                import mlflow
                mlflow.log_artifact(artifact_path)
            except Exception as e:
                logger.debug(f"MLflow artifact logging failed: {e}")

    def log_evaluation_results(self, results: Dict):
        """
        Log evaluation results from DeepfakeEvaluator.

        Args:
            results: Evaluation results dictionary
        """
        # Extract key metrics
        overall = results.get('overall', {})
        metrics_to_log = {
            'eval_accuracy': overall.get('accuracy', 0),
            'eval_auc_roc': overall.get('auc_roc', 0),
            'eval_eer': overall.get('eer', 0),
            'eval_precision_fake': overall.get('precision_fake', 0),
            'eval_recall_fake': overall.get('recall_fake', 0),
            'eval_f1_fake': overall.get('f1_fake', 0)
        }

        self.log_metrics(metrics_to_log)

        # Log per-manipulation type results
        manip_metrics = results.get('by_manipulation_type', {})
        for manip_type, metrics in manip_metrics.items():
            manip_prefix = f"eval_{manip_type}"
            manip_log = {
                f"{manip_prefix}_auc": metrics.get('auc_roc', 0),
                f"{manip_prefix}_accuracy": metrics.get('accuracy', 0),
                f"{manip_prefix}_count": metrics.get('count', 0)
            }
            self.log_metrics(manip_log)

    def log_training_step(
        self,
        epoch: int,
        loss: float,
        learning_rate: float = None,
        batch_idx: int = None
    ):
        """
        Log training step metrics.

        Args:
            epoch: Current epoch
            loss: Training loss
            learning_rate: Current learning rate
            batch_idx: Batch index (for per-batch logging)
        """
        step = epoch
        metrics = {'train_loss': loss}

        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate

        if batch_idx is not None:
            step = epoch * 100 + batch_idx  # Combine epoch and batch

        self.log_metrics(metrics, step=step)

    def log_validation_step(
        self,
        epoch: int,
        loss: float,
        accuracy: float = None,
        auc: float = None
    ):
        """
        Log validation step metrics.

        Args:
            epoch: Current epoch
            loss: Validation loss
            accuracy: Validation accuracy
            auc: Validation AUC-ROC
        """
        metrics = {'val_loss': loss}

        if accuracy is not None:
            metrics['val_accuracy'] = accuracy

        if auc is not None:
            metrics['val_auc'] = auc

        self.log_metrics(metrics, step=epoch)

    def end_run(self, status: str = "completed"):
        """
        End the experiment run.

        Args:
            status: Final status ("completed", "failed", "aborted")
        """
        if self.wandb_run:
            try:
                import wandb
                self.wandb_run.finish()
            except Exception as e:
                logger.debug(f"W&B finish failed: {e}")

        if self.mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception as e:
                logger.debug(f"MLflow finish failed: {e}")

        logger.info(f"Experiment run ended with status: {status}")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_run_url(self) -> Optional[str]:
        """Get URL to view this run online."""
        if self.wandb_run:
            try:
                return self.wandb_run.get_url()
            except Exception as e:
                logger.debug(f"Could not get W&B URL: {e}")

        return None


class LocalExperimentLogger:
    """
    Local file-based experiment logging (no cloud dependency).
    Useful for offline experiments or as fallback when cloud services unavailable.
    """

    def __init__(
        self,
        experiment_name: str = "experiment",
        log_dir: str = "logs"
    ):
        """
        Initialize local logger.

        Args:
            experiment_name: Name of experiment
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = self.log_dir / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.params_file = self.run_dir / "params.json"
        self.metadata_file = self.run_dir / "metadata.json"

        logger.info(f"Local experiment logger initialized at {self.run_dir}")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=2)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to JSONL file."""
        entry = {'step': step} if step is not None else {}
        entry.update(metrics)

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_metadata(self, metadata: Dict):
        """Log experiment metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_metrics_dataframe(self):
        """
        Load logged metrics as pandas DataFrame.

        Returns:
            pandas.DataFrame with logged metrics
        """
        try:
            import pandas as pd
            lines = []
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    lines.append(json.loads(line))
            return pd.DataFrame(lines)
        except ImportError:
            logger.warning("pandas not available for DataFrame conversion")
            return None


def create_tracker(
    backend: str = "local",
    project_name: str = "deepfake-detection",
    run_name: str = "experiment",
    **kwargs
) -> Any:
    """
    Factory function to create appropriate tracker.

    Args:
        backend: "wandb", "mlflow", "local", or "both"
        project_name: Project name
        run_name: Run name
        **kwargs: Additional arguments

    Returns:
        Tracker instance (ExperimentTracker or LocalExperimentLogger)
    """
    if backend == "local":
        return LocalExperimentLogger(
            experiment_name=run_name,
            log_dir=kwargs.get('log_dir', 'logs')
        )
    else:
        return ExperimentTracker(
            project_name=project_name,
            run_name=run_name,
            backend=backend,
            offline=kwargs.get('offline', False)
        )
