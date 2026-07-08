"""
Abstract base class for all evaluators.

Defines the interface that every evaluator must implement. Each
evaluator loads a trained model, runs it against a validation/test
set, and collects all relevant metrics for benchmarking.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from configs.config_loader import RunConfig
from models.base_model import BaseModel


class BaseEvaluator(ABC):
    """
    Abstract evaluator that benchmarks a trained model.

    Subclasses implement evaluate(), which loads model weights, runs
    inference on the test set, and returns a dict of all metrics.
    """

    def __init__(self, model: BaseModel, config: RunConfig):
        self._model = model
        self._config = config

    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def config(self) -> RunConfig:
        return self._config

    @abstractmethod
    def evaluate(
        self, weights_path: str | Path | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Run full evaluation and return all metrics.

        Args:
            weights_path: Path to trained .pt weights.
                          If None, resolves from the config run_id.
            **kwargs: Override evaluation parameters.

        Returns:
            Dict with keys like:
                - mAP50, mAP50-95
                - precision, recall, f1
                - inference_speed (ms per image)
                - model_params, model_gflops
                - model_size_mb
        """
        ...
