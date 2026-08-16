import contextlib
import sys
import types
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from configs.config_loader import RunConfig
from models.base_model import BaseModel
from training.tiny_train import TinyTrain
from pipeline_utils.logging_utils import get_logger

_log = get_logger("pipeline")

YOLOV5_PATH = str(Path(__file__).parents[2] / "yolov5")
_REPO_MODELS_INIT = str(Path(__file__).parents[2] / "models" / "__init__.py")

# Keep yolov5/ on sys.path permanently so multiprocessing worker processes
# can find `utils.*` modules when unpickling YOLOv5 dataset objects.
# We manage the repo-root models/ vs yolov5/models/ naming conflict
# dynamically in the context manager below.
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)


@contextlib.contextmanager
def _yolov5_import_context():
    """Promote yolov5/ to front of sys.path for imports, handling
    the repo-root models/ vs yolov5/models/ naming conflict.
    Restores sys.path position and sys.modules on exit."""
    saved_models = sys.modules.pop("models", None)

    # Move YOLOV5_PATH to front during import
    if YOLOV5_PATH in sys.path:
        sys.path.remove(YOLOV5_PATH)
    sys.path.insert(0, YOLOV5_PATH)

    try:
        yield
    finally:
        # Move YOLOV5_PATH back to end (still on sys.path for workers)
        if YOLOV5_PATH in sys.path:
            sys.path.remove(YOLOV5_PATH)
        sys.path.append(YOLOV5_PATH)

        # Restore original repo-root models/, evict yolov5 models/ if loaded
        if saved_models is not None:
            sys.modules["models"] = saved_models
        elif "models" in sys.modules:
            loaded = sys.modules["models"]
            if getattr(loaded, "__file__", None) != _REPO_MODELS_INIT:
                del sys.modules["models"]


def _import_yolov5(module_name: str):
    """Import a top-level module from the yolov5/ directory."""
    with _yolov5_import_context():
        return __import__(module_name)


def _resolve_torch_device(device):
    """Normalize a device string (e.g. '0' -> 'cuda:0') for torch .to()/.cuda()."""
    if isinstance(device, str) and device.strip().isdigit():
        return f"cuda:{device.strip()}"
    return device


def _load_yolov5_model(path: str | Path, device="cpu"):
    """Load a YOLOv5-format checkpoint via the yolov5 repo's own loader.

    ultralytics refuses YOLOv5 checkpoints (they pickle 'models.yolo'), so the
    yolov5 family must load/train/val through the vendored yolov5 repository.
    Returns a lightweight object exposing `.model` (the DetectionModel) and
    `.ckpt_path` (used for locating weights at val time).
    """
    with _yolov5_import_context():
        from models.experimental import attempt_load
        model = attempt_load(str(path), device=_resolve_torch_device(device), fuse=False)
    model.eval()
    return types.SimpleNamespace(model=model, ckpt_path=str(path))


class YOLODetector(BaseModel):

    def __init__(self, config: RunConfig):
        super().__init__(config)
        self._model: YOLO | None = None
        self._last_freeze_list: list | None = None

    @property
    def model(self) -> YOLO | None:
        return self._model

    @property
    def last_freeze_list(self) -> list | None:
        """Nested freeze list computed by TinyTrain during the last train() call."""
        return self._last_freeze_list

    def load(self, weights_path: str | Path | None = None):
        path = weights_path or f"{self.family}{self.variant}.pt"
        if self.family == "yolov5":
            self._model = _load_yolov5_model(str(path), device="cpu")
        else:
            self._model = YOLO(str(path))

    @staticmethod
    def _single(v):
        return v[0] if isinstance(v, (list, tuple)) else v

    def _build_yolov5_train_args(self, override_kwargs: dict) -> dict:
        hp = self._config.hyperparameters
        tr = self._config.training
        inf = self._config.inference
        ds = self._config.dataset

        args = {
            "weights": f"{self.family}{self.variant}.pt",
            "data": ds.get("data_yaml", "./dataset/data.yaml"),
            "cfg": f"{YOLOV5_PATH}/models/{self.family}{self.variant}.yaml",
            "epochs": tr.get("epochs", 100),
            "batch_size": tr.get("batch", 16),
            "imgsz": self._single(inf.get("imgsz", 640)),
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
            "cos_lr": tr.get("cos_lr", False),
            "seed": tr.get("seed", 42),
            "single_cls": tr.get("single_cls", False),
            "label_smoothing": tr.get("label_smoothing", 0.0),
            "save_period": tr.get("save_period", -1),
            "exist_ok": True,
            "project": tr.get("project", "runs/train"),
            "name": tr.get("name", self._config.run_id),
            "nosave": False,
            "noval": True,
            "noplots": False,
            "multi_scale": tr.get("multi_scale", False),
            "freeze": [0],
        }

        args.update(override_kwargs)
        return args

    def train(self, **override_kwargs) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        family = self._config.family
        ds = self._config.dataset

        data_yaml = override_kwargs.pop("data", ds.get("data_yaml", "./dataset/data.yaml"))
        weights_path = override_kwargs.pop("weights", None)
        tiny_train_enabled = override_kwargs.pop("tiny_train_enabled", None)

        freeze_ratio = override_kwargs.pop("freeze_ratio", None)
        tt_config = self._config.get("tiny_train") or self._config.global_config.get("tiny_train", {})
        if freeze_ratio is not None:
            tt_config = {**tt_config, "freeze_portion": freeze_ratio}
        if tiny_train_enabled is not None:
            tt_config = {**tt_config, "enabled": bool(tiny_train_enabled)}

        if family == "yolov5":
            train_args = self._build_yolov5_train_args(override_kwargs)
            train_args["data"] = data_yaml
            if weights_path is not None:
                train_args["weights"] = str(weights_path)
            _log.debug("YOLOv5 train args: freeze=%s (type=%s)",
                       train_args.get("freeze"), type(train_args.get("freeze")).__name__)

            if tt_config.get("enabled", False):
                _log.info("TinyTrain enabled for %s (freeze_portion=%.2f)",
                          self._config.run_id, tt_config.get("freeze_portion", 0.5))
                tt = TinyTrain(tt_config)
                freeze_list = tt.apply(
                    variant=self._config.variant,
                    data_yaml=data_yaml,
                    device=str(train_args.get("device", "cpu")),
                    imgsz=train_args.get("imgsz", 640),
                    family="yolov5",
                    weights_path=weights_path,
                )
                self._last_freeze_list = freeze_list
                _log.info("TinyTrain returned freeze list with %d entries", len(freeze_list))
                if freeze_list:
                    train_args["freeze"] = freeze_list
                    _log.info("OVERRIDE freeze with TinyTrain nested list (%d layer groups)",
                              len(freeze_list))
            else:
                self._last_freeze_list = None

            yolov5_train = _import_yolov5("train")

            result = yolov5_train.run(**train_args)
            return result

        hp = self._config.hyperparameters
        tr = self._config.training

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
            "cos_lr": tr.get("cos_lr", False),
            "seed": tr.get("seed", 42),
            "amp": tr.get("amp", False),
            "single_cls": tr.get("single_cls", False),
            "label_smoothing": tr.get("label_smoothing", 0.0),
            "freeze": tr.get("freeze", 0),
            "save_period": tr.get("save_period", -1),
            "exist_ok": tr.get("exist_ok", False),
            "project": tr.get("project", "runs/train"),
            "name": tr.get("name", self._config.run_id),
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
            "resume": tr.get("resume", False),
            "multi_scale": tr.get("multi_scale", False),
        }

        if family == "yolov8":
            args.update({
                "dfl": hp.get("dfl", 1.5),
                "overlap_mask": tr.get("overlap_mask", True),
                "mask_ratio": tr.get("mask_ratio", 4),
                "dropout": tr.get("dropout", 0.0),
            })

        args.update(override_kwargs)

        if family == "yolov8":
            _log.debug("YOLOv8 args before TinyTrain: freeze=%s (type=%s)",
                       args.get("freeze"), type(args.get("freeze")).__name__)
            if tt_config.get("enabled", False):
                _log.info("TinyTrain enabled for %s (freeze_portion=%.2f)",
                          self._config.run_id, tt_config.get("freeze_portion", 0.5))
                tt = TinyTrain(tt_config)
                freeze_list = tt.apply(
                    variant=self._config.variant,
                    data_yaml=data_yaml,
                    device=str(args.get("device", "cpu")),
                    imgsz=self._config.inference.get("imgsz", 640),
                    family="yolov8",
                    weights_path=weights_path,
                )
                self._last_freeze_list = freeze_list
                _log.info("TinyTrain returned freeze list with %d entries", len(freeze_list))
                if freeze_list:
                    args["freeze"] = freeze_list
                    _log.info("OVERRIDE freeze with TinyTrain nested list (%d layer groups)",
                              len(freeze_list))
            else:
                self._last_freeze_list = None

        if family == "yolov8":
            args["project"] = str(Path(args["project"]).resolve())

        _log.info("Starting %s train() with freeze=%s",
                  family, args.get("freeze", "not set"))
        results = self._model.train(**args)

        return results

    def predict(self, source: str | Path | list[str], **kwargs) -> Any:
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
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        family = self._config.family

        if family == "yolov5":
            yolov5_val = _import_yolov5("val")

            ds = self._config.dataset
            ev = self._config.global_config.get("evaluation", {})
            tr = self._config.training

            val_args = {
                "weights": kwargs.pop("weights", str(self._model.ckpt_path)),
                "data": ds.get("data_yaml", "./dataset/data.yaml"),
                "imgsz": self._single(self._config.inference.get("imgsz", 640)),
                "batch_size": ev.get("batch", 16),
                "task": ev.get("split", "test"),
                "half": ev.get("half", False),
                "device": tr.get("device", "cpu"),
                "workers": 0,
                "exist_ok": True,
                "save_json": False,
                "save_txt": False,
                "plots": False,
            }
            val_args.update(kwargs)
            results, maps, times = yolov5_val.run(**val_args)
            mp, mr, map50, map, *losses = results
            return {
                "mp": mp, "mr": mr, "map50": map50, "map50_95": map,
                "maps": maps, "speed": times, "losses": losses,
            }

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
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._model.save(str(path))

    def export(self, format: str = "onnx", path: str | Path | None = None) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.export(format=format)
