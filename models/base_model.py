from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from configs.config_loader import RunConfig


class BaseModel(ABC):
    def __init__(self, config: RunConfig):
        self._config = config

    @property
    def run_id(self) -> str:
        return self._config.run_id

    @property
    def family(self) -> str:
        return self._config.family

    @property
    def variant(self) -> str:
        return self._config.variant

    @property
    def config(self) -> RunConfig:
        return self._config

    @abstractmethod
    def load(self, weights_path: str | Path | None = None):
        ...

    @abstractmethod
    def train(self, **override_kwargs) -> Any:
        ...

    @abstractmethod
    def predict(self, source: str | Path | list[str], **kwargs) -> Any:
        ...

    @abstractmethod
    def val(self, **kwargs) -> Any:
        ...

    @abstractmethod
    def save(self, path: str | Path):
        ...

    @abstractmethod
    def export(self, format: str = "onnx", path: str | Path | None = None) -> str:
        ...
