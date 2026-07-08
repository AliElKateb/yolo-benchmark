from configs.config_loader import RunConfig
from models.base_model import BaseModel
from models.detection.yolo_detector import YOLODetector


def create_model(config: RunConfig) -> BaseModel:
    family = config.family

    if family in ("yolov5", "yolov8"):
        return YOLODetector(config)

    raise ValueError(f"Unknown model family: {family}")
