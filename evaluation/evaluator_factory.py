"""
Factory for creating the appropriate evaluator for a given model.

Maps model families (yolov5, yolov8) to their corresponding evaluators.
"""

from configs.config_loader import RunConfig
from models.base_model import BaseModel
from evaluation.base_evaluation import BaseEvaluator
from evaluation.detector.yolo_evaluator import YOLOEvaluator


def create_evaluator(model: BaseModel, config: RunConfig) -> BaseEvaluator:
    """
    Create the evaluator matching the model's family.

    Args:
        model: An instantiated model (e.g. YOLODetector).
        config: The run configuration.

    Returns:
        A concrete BaseEvaluator subclass ready to run evaluation.

    Raises:
        ValueError: If the model family has no registered evaluator.
    """
    family = config.family

    if family in ("yolov5", "yolov8"):
        return YOLOEvaluator(model, config)

    raise ValueError(f"No evaluator registered for model family: {family}")
