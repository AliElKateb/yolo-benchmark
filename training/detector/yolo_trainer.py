"""
Trainer for YOLOv5/YOLOv8 detection models.

Wraps a YOLODetector instance and runs training using the parameters
defined in the run configuration. Handles model loading, dataset
validation, training execution, and result collection.
"""

from typing import Any

from configs.config_loader import RunConfig
from models.base_model import BaseModel
from training.base_trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):
    """
    Trainer for YOLO detection models.

    Delegates the actual training call to the model's train() method
    while adding pre-flight checks and result reporting.
    """

    def __init__(self, model: BaseModel, config: RunConfig):
        super().__init__(model, config)

    def train(self, **kwargs) -> Any:
        """
        Execute the full training pipeline for this YOLO run.

        Steps:
            1. Validate that the dataset exists.
            2. Load pretrained weights.
            3. Run training with config parameters.
            4. Log and return results.

        Args:
            **kwargs: Overrides for any training parameter.

        Returns:
            Training results dict with final metrics.
        """
        run_id = self._config.run_id
        print(f"\n{'='*60}")
        print(f"  Starting training: {run_id}")
        print(f"  Model: {self._config.family} {self._config.variant}")
        print(f"{'='*60}")

        if not self.validate_dataset():
            raise FileNotFoundError(
                f"Cannot train {run_id}: dataset not found."
            )

        print(f"  Loading weights for {run_id} ...")
        self._model.load()
        print(f"  Training {run_id} ...")
        results = self._model.train(**kwargs)

        print(f"  Training complete: {run_id}")
        return results
