import yaml
from pathlib import Path
from typing import Optional


class DetectorConfig:
    def __init__(self, config_path: str | Path):
        with open(config_path) as f:
            self._data = yaml.safe_load(f)
        self._runs = [RunConfig(r, parent=self) for r in self._data.get("runs", [])]

    @property
    def runs(self) -> list["RunConfig"]:
        return self._runs

    @property
    def enabled_runs(self) -> list["RunConfig"]:
        return [r for r in self._runs if r.enabled]

    def get_run(self, run_id: str) -> Optional["RunConfig"]:
        for r in self._runs:
            if r.run_id == run_id:
                return r
        return None

    def runs_by_family(self, family: str) -> list["RunConfig"]:
        return [r for r in self._runs if r.family == family]

    def enabled_runs_by_family(self, family: str) -> list["RunConfig"]:
        return [r for r in self.enabled_runs if r.family == family]

    @property
    def global_config(self) -> dict:
        """Top-level settings (evaluation, etc.), exposed publicly."""
        return {k: v for k, v in self._data.items() if k != "runs"}


class RunConfig:
    def __init__(self, data: dict, parent: Optional["DetectorConfig"] = None):
        self._data = data
        self._parent = parent

    @property
    def run_id(self) -> str:
        return self._data.get("run_id", "")

    @property
    def enabled(self) -> bool:
        return self._data.get("enabled", False)

    @property
    def family(self) -> str:
        return self._data.get("model", {}).get("family", "")

    @property
    def variant(self) -> str:
        return self._data.get("model", {}).get("variant", "")

    @property
    def task(self) -> str:
        return self._data.get("model", {}).get("task", "detect")

    @property
    def model(self) -> dict:
        return self._data.get("model", {})

    @property
    def dataset(self) -> dict:
        return self._data.get("dataset", {})

    @property
    def hyperparameters(self) -> dict:
        return self._data.get("hyperparameters", {})

    @property
    def training(self) -> dict:
        return self._data.get("training", {})

    @property
    def inference(self) -> dict:
        return self._data.get("inference", {})

    @property
    def metadata(self) -> dict:
        return self._data.get("metadata", {})

    @property
    def global_config(self) -> dict:
        """Access top-level config sections (evaluation, etc.)."""
        return self._parent.global_config if self._parent else {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"RunConfig(run_id='{self.run_id}', enabled={self.enabled})"


CONFIG_PATH = Path(__file__).parent / "detector_config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> DetectorConfig:
    return DetectorConfig(config_path)
