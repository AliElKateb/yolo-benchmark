"""
Abstract base class for all trainers.

Defines the interface that every trainer must implement. Each trainer
wraps a model and a run configuration, orchestrating the full training
lifecycle: load, validate, train, and collect results.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from configs.config_loader import RunConfig
from models.base_model import BaseModel


class BaseTrainer(ABC):
    """
    Abstract trainer that orchestrates model training.

    Subclasses need to implement the train() method, which runs the
    actual training loop. The base provides common validation and
    reporting infrastructure.
    """

    def __init__(self, model: BaseModel, config: RunConfig):
        self._model = model
        self._config = config

    @property
    def model(self) -> BaseModel:
        """The model instance being trained."""
        return self._model

    @property
    def config(self) -> RunConfig:
        """The run configuration dictating how training proceeds."""
        return self._config

    @abstractmethod
    def train(self, **kwargs) -> Any:
        """
        Run the full training pipeline.

        Should handle loading weights, running training, and
        returning results/metrics.

        Args:
            **kwargs: Overrides for any training parameter.

        Returns:
            Training results dict with final and per-epoch metrics.
        """
        ...

    def validate_dataset(self) -> bool:
        """
        Check that the dataset's data.yaml exists before training.

        Returns:
            True if the dataset yaml was found, False otherwise.
        """
        data_yaml = self._config.dataset.get("data_yaml", "")
        if not data_yaml:
            print("  [WARN] No data_yaml specified in config.")
            return False
        exists = Path(data_yaml).exists()
        if not exists:
            print(f"  [ERROR] data_yaml not found: {data_yaml}")
        return exists
