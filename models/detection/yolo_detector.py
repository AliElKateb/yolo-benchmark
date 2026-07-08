"""
YOLOv5/YOLOv8 detector implementation wrapping Ultralytics.

Maps configuration from detector_config.yaml to the Ultralytics YOLO API,
handling both model families and all size variants (n/s/m/l/x).
"""

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from configs.config_loader import RunConfig
from models.base_model import BaseModel


class YOLODetector(BaseModel):
    """
    YOLO object detector that wraps Ultralytics' YOLO class.

    Supports both YOLOv5 and YOLOv8 across all size variants. Configuration
    (hyperparameters, training settings, inference parameters) is loaded from
    the run's RunConfig object rather than being hardcoded.
    """

    def __init__(self, config: RunConfig):
        super().__init__(config)
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO | None:
        """Access the underlying Ultralytics YOLO instance after load()."""
        return self._model

    def load(self, weights_path: str | Path | None = None):
        """
        Load pretrained weights into the model.

        Args:
            weights_path: Path to .pt weights file.
                          Defaults to e.g. 'yolov8n.pt' based on family+variant.
        """
        path = weights_path or f"{self.family}{self.variant}.pt"
        self._model = YOLO(str(path))

    def train(self, **override_kwargs) -> dict[str, Any] | None:
        """
        Train the model using settings from the run configuration.

        Builds a kwargs dict from the run's hyperparameters, training, and
        dataset sections, then passes everything to Ultralytics' model.train().
        Additional kwargs override any config values.

        Returns:
            Training results dict with metrics per epoch.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        hp = self._config.hyperparameters
        tr = self._config.training
        ds = self._config.dataset

        # Point Ultralytics to the dataset's data.yaml
        data_yaml = ds.get("data_yaml", "./dataset/data.yaml")

        args = {
            "data": data_yaml,
            "epochs": tr.get("epochs", 100),
            "batch": tr.get("batch", 16),
            "imgsz": self._config.inference.get("imgsz", 640),
            "patience": tr.get("early_stopping", {}).get("patience", 20),
            "device": tr.get("device", "cpu"),
            "workers": tr.get("workers", 8),
            "optimizer": tr.get("optimizer", "SGD"),
            "lr0": hp.get("lr", 0.01),
            "lrf": hp.get("lrf", 0.01),
            "momentum": hp.get("momentum", 0.937),
            "weight_decay": hp.get("weight_decay", 0.0005),
            "warmup_epochs": hp.get("warmup_epochs", 3.0),
            "warmup_momentum": hp.get("warmup_momentum", 0.8),
            "warmup_bias_lr": hp.get("warmup_bias_lr", 0.1),
            "box": hp.get("box", 7.5),
            "cls": hp.get("cls", 0.5),
            "dfl": hp.get("dfl", 1.5),
            "cos_lr": tr.get("cos_lr", False),
            "seed": tr.get("seed", 42),
            "pretrained": tr.get("pretrained", True),
            "amp": tr.get("amp", False),
            "single_cls": tr.get("single_cls", False),
            "label_smoothing": tr.get("label_smoothing", 0.0),
            "freeze": tr.get("freeze", 0),
            "save_period": tr.get("save_period", -1),
            "exist_ok": tr.get("exist_ok", False),
            "project": tr.get("project", "runs/train"),
            "name": tr.get("name", self.run_id),
            "deterministic": tr.get("deterministic", True),
            "cache": tr.get("cache", False),
            "hsv_h": hp.get("hsv_h", 0.015),
            "hsv_s": hp.get("hsv_s", 0.7),
            "hsv_v": hp.get("hsv_v", 0.4),
            "degrees": hp.get("degrees", 0.0),
            "translate": hp.get("translate", 0.1),
            "scale": hp.get("scale", 0.5),
            "shear": hp.get("shear", 0.0),
            "perspective": hp.get("perspective", 0.0),
            "flipud": hp.get("flipud", 0.0),
            "fliplr": hp.get("fliplr", 0.5),
            "mosaic": hp.get("mosaic", 1.0),
            "mixup": hp.get("mixup", 0.0),
            "copy_paste": hp.get("copy_paste", 0.0),
        }

        args.update(override_kwargs)
        results = self._model.train(**args)
        return results

    def predict(self, source: str | Path | list[str], **kwargs) -> Any:
        """
        Run inference on images.

        Args:
            source: Path to image, directory, video, or list of paths.
            **kwargs: Override any inference setting from the config.

        Returns:
            List of Ultralytics Results objects (one per image).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inf = self._config.inference

        args = {
            "source": source,
            "imgsz": inf.get("imgsz", 640),
            "conf": inf.get("conf_thres", 0.25),
            "iou": inf.get("iou_thres", 0.45),
            "max_det": inf.get("max_det", 300),
            "device": self._config.training.get("device", "cpu"),
            "half": inf.get("half", False),
            "augment": inf.get("augment", False),
            "visualize": inf.get("visualize", False),
        }

        args.update(kwargs)
        return self._model.predict(**args)

    def val(self, **kwargs) -> Any:
        """
        Run validation on the dataset's test/val split.

        Uses the data.yaml from the run config to locate the validation
        images and labels. Returns a DetMetrics object with all metrics.

        Args:
            **kwargs: Override any validation parameter (batch, imgsz, conf, etc.).

        Returns:
            Ultralytics DetMetrics object with mAP, precision, recall, etc.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        ev = self._config.global_config.get("evaluation", {})
        ds = self._config.dataset
        tr = self._config.training

        args = {
            "data": ds.get("data_yaml", "./dataset/data.yaml"),
            "batch": ev.get("batch", 16),
            "imgsz": self._config.inference.get("imgsz", 640),
            "conf": ev.get("conf_thres", 0.25),
            "iou": ev.get("iou_thres", 0.45),
            "half": ev.get("half", False),
            "device": tr.get("device", "cpu"),
            "split": ev.get("split", "test"),
        }

        args.update(kwargs)
        return self._model.val(**args)

    def save(self, path: str | Path):
        """Save trained model weights to disk."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._model.save(str(path))

    def export(self, format: str = "onnx", path: str | Path | None = None) -> str:
        """Export the model to the specified format (ONNX, TorchScript, etc.)."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.export(format=format)
