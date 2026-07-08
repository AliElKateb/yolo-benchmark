"""
Factory for creating the appropriate trainer for a given model.

Maps model families (yolov5, yolov8) to their corresponding trainers.
"""

from configs.config_loader import RunConfig
from models.base_model import BaseModel
from training.base_trainer import BaseTrainer
from training.detector.yolo_trainer import YOLOTrainer


def create_trainer(model: BaseModel, config: RunConfig) -> BaseTrainer:
    """
    Create the trainer matching the model's family.

    Args:
        model: An instantiated model (e.g. YOLODetector).
        config: The run configuration driving training.

    Returns:
        A concrete BaseTrainer subclass ready to run training.

    Raises:
        ValueError: If the model family has no registered trainer.
    """
    family = config.family

    if family in ("yolov5", "yolov8"):
        return YOLOTrainer(model, config)

    raise ValueError(f"No trainer registered for model family: {family}")
