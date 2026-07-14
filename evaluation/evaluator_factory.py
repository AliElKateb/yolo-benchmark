"""
Factory for creating the appropriate evaluator for a given model.

Maps model families (yolov5, yolov8) to their corresponding evaluators.
"""

from configs.config_loader import RunConfig
from models.base_model import BaseModel
from evaluation.base_evaluation import BaseEvaluator
from evaluation.detector.yolo_evaluator import YOLOEvaluator


def create_evaluator(model: BaseModel, config: RunConfig, experiment_name: str | None = None) -> BaseEvaluator:
    family = config.family

    if family in ("yolov5", "yolov8"):
        return YOLOEvaluator(model, config, experiment_name=experiment_name)

    raise ValueError(f"No evaluator registered for model family: {family}")
