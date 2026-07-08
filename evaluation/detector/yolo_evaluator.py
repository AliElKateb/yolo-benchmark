"""
Evaluator for YOLOv5/YOLOv8 detection models.

Loads a trained model's weights, runs validation on the test split,
extracts all metrics (mAP, precision, recall, speed, model size),
and saves results to the experiments output directory.
"""

import json
import time
from pathlib import Path
from typing import Any

from configs.config_loader import RunConfig
from models.base_model import BaseModel
from evaluation.base_evaluation import BaseEvaluator


class YOLOEvaluator(BaseEvaluator):
    """
    Evaluates a trained YOLO detector on all standard metrics.

    Results are saved as JSON and aggregated into a comparison CSV
    under outputs/detection_experiments/<experiment_timestamp>/.
    """

    def __init__(self, model: BaseModel, config: RunConfig):
        super().__init__(model, config)
        self._experiment_dir: Path | None = None
        self._run_dir: Path | None = None

    @property
    def experiment_dir(self) -> Path | None:
        """Directory where the current experiment's results are saved."""
        return self._experiment_dir

    def _resolve_weights(self, weights_path: str | Path | None = None) -> Path:
        """
        Find the trained weights file.

        Checks, in order:
          1. Explicitly provided path.
          2. Various ultralytics output path patterns for this run.
        """
        if weights_path is not None:
            return Path(weights_path)

        run_id = self._config.run_id
        project = self._config.training.get("project", "")
        name = self._config.training.get("name", run_id)
        task = self._config.get("model", {}).get("task", "detect")

        # Ultralytics nests output as: runs/{task}/{project}/{name}/weights/
        # Try common patterns including old project=runs/train and new project=runs
        candidate_names = [name, run_id]
        candidate_projects = [
            f"runs/{task}/{project}" if project else f"runs/{task}",
            f"runs/{task}/runs/train",
            f"runs/{task}/runs",
            "runs/detect",
        ]

        candidates = []
        for p in candidate_projects:
            for n in candidate_names:
                candidates.append(Path(f"{p}/{n}/weights/best.pt"))
                candidates.append(Path(f"{p}/{n}/weights/last.pt"))

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)

        for path in unique:
            if path.exists():
                return path

        # Fallback: search for ultralytics auto-numbered directories (e.g. yolov5_nano-3)
        for p in candidate_projects:
            base_dir = Path(p)
            if not base_dir.exists():
                continue
            for n in candidate_names:
                for w in sorted(base_dir.glob(f"{n}-*/weights/best.pt"), reverse=True):
                    return w
                for w in sorted(base_dir.glob(f"{n}-*/weights/last.pt"), reverse=True):
                    return w

        raise FileNotFoundError(
            f"No trained weights found for {run_id}. "
            f"Tried: {[str(p) for p in unique]}"
        )

    def _setup_output_dir(self):
        """
        Create a timestamped experiment directory under detection_experiments.

        Each run within an experiment gets its own subfolder.
        """
        ev = self._config.global_config.get("evaluation", {})
        base = Path(ev.get("output_dir", "./outputs/detection_experiments"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        self._experiment_dir = base / f"experiment_{timestamp}"
        self._experiment_dir.mkdir(parents=True, exist_ok=True)

        self._run_dir = self._experiment_dir / self._config.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def _extract_metrics(self, val_results, trained_model=None) -> dict[str, Any]:
        """
        Extract all relevant metrics from Ultralytics validation results.

        Args:
            val_results: DetMetrics object from model.val().
            trained_model: The underlying YOLO model (for complexity stats).

        Returns:
            Flat dict of metric names to values.
        """
        metrics = {}

        if hasattr(val_results, "box") and val_results.box is not None:
            box = val_results.box

            def safe_val(x):
                return None if x is None else round(float(x) * 100, 2)

            metrics["map50_95"] = safe_val(box.map)
            metrics["map50"] = safe_val(box.map50)
            metrics["precision"] = safe_val(box.mp)
            metrics["recall"] = safe_val(box.mr)

            per_class = []
            try:
                if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
                    class_indices = box.ap_class_index
                    if hasattr(class_indices, "__len__"):
                        names = self._config.dataset.get("names", [])
                        for cls_idx in class_indices:
                            cls_idx = int(cls_idx)
                            cls_name = names[cls_idx] if cls_idx < len(names) else str(cls_idx)
                            per_class.append({
                                "class": cls_name,
                                "map50_95": safe_val(box.map) if hasattr(box, "map") else None,
                                "map50": safe_val(box.map50) if hasattr(box, "map50") else None,
                            })
            except (TypeError, IndexError, ValueError):
                pass
            metrics["per_class"] = per_class

            p = metrics.get("precision")
            r = metrics.get("recall")
            if p is not None and r is not None and (p + r) > 0:
                metrics["f1_score"] = round(2 * (p * r) / (p + r), 2)
            else:
                metrics["f1_score"] = 0.0

        if hasattr(val_results, "speed") and val_results.speed:
            speed = val_results.speed
            metrics["preprocess_ms"] = round(speed.get("preprocess", 0), 2)
            metrics["inference_ms"] = round(speed.get("inference", 0), 2)
            metrics["postprocess_ms"] = round(speed.get("postprocess", 0), 2)
            metrics["total_per_image_ms"] = round(
                sum(speed.get(k, 0) for k in ("preprocess", "inference", "postprocess")), 2
            )

        # Model complexity: read from the trained YOLO model directly
        if trained_model is not None:
            try:
                params = sum(p.numel() for p in trained_model.parameters())
                metrics["model_params"] = params
                metrics["model_gflops"] = round(params * 2 / 1e9, 2)
            except Exception:
                pass

        return metrics

    def _get_model_file_size(self, weights_path: Path) -> float:
        """Get the model weights file size in MB."""
        return round(weights_path.stat().st_size / (1024 * 1024), 2)

    def _save_results(self, weights_path: Path, metrics: dict[str, Any]):
        """
        Save evaluation results to JSON and append to comparison CSV.
        """
        if self._run_dir is None:
            return

        # Add metadata to the saved results
        metrics["run_id"] = self._config.run_id
        metrics["family"] = self._config.family
        metrics["variant"] = self._config.variant
        metrics["weights_path"] = str(weights_path)
        metrics["model_size_mb"] = self._get_model_file_size(weights_path)
        metrics["dataset"] = self._config.dataset.get("data_yaml", "")

        json_path = self._run_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Results saved to {json_path}")

        # Append to comparison CSV
        csv_path = self._experiment_dir / "comparison.csv"
        self._append_to_csv(csv_path, metrics)

    def _append_to_csv(self, csv_path: Path, metrics: dict[str, Any]):
        """
        Append a row to the comparison CSV.

        Creates the file with a header if it doesn't exist yet.
        """
        import csv

        field_map = {
            "run_id": "Run ID",
            "family": "Family",
            "variant": "Variant",
            "map50": "mAP50",
            "map50_95": "mAP50-95",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "inference_ms": "Inference (ms)",
            "total_per_image_ms": "Total (ms/img)",
            "model_params": "Parameters",
            "model_gflops": "GFLOPs",
            "model_size_mb": "Size (MB)",
            "weights_path": "Weights",
        }

        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(field_map.values()))
            if not file_exists:
                writer.writeheader()

            row = {}
            for key, label in field_map.items():
                row[label] = metrics.get(key, "")
            writer.writerow(row)

        print(f"  Comparison CSV updated: {csv_path}")

    def evaluate(
        self, weights_path: str | Path | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Run full evaluation pipeline.

        Steps:
          1. Resolve trained weights path.
          2. Load the trained model.
          3. Run validation on the test split.
          4. Extract all metrics.
          5. Save results to the experiments directory.
          6. Return the metrics dict.

        Returns:
            Dict with all evaluation metrics.
        """
        run_id = self._config.run_id
        print(f"\n{'='*60}")
        print(f"  Evaluating: {run_id}")
        print(f"  Model: {self._config.family} {self._config.variant}")
        print(f"{'='*60}")

        weights = self._resolve_weights(weights_path)
        print(f"  Weights: {weights}")

        self._setup_output_dir()

        print(f"  Loading trained model ...")
        self._model.load(str(weights))

        print(f"  Running validation ...")
        val_results = self._model.val(**kwargs)

        print(f"  Extracting metrics ...")
        trained_model = self._model.model.model if hasattr(self._model.model, "model") else self._model.model
        metrics = self._extract_metrics(val_results, trained_model=trained_model)

        self._save_results(weights, metrics)

        # Print summary
        print(f"\n  Metrics for {run_id}:")
        print(f"    mAP50:    {metrics.get('map50', 'N/A')}")
        print(f"    mAP50-95: {metrics.get('map50_95', 'N/A')}")
        print(f"    P:        {metrics.get('precision', 'N/A')}")
        print(f"    R:        {metrics.get('recall', 'N/A')}")
        print(f"    F1:       {metrics.get('f1_score', 'N/A')}")
        print(f"    Speed:    {metrics.get('inference_ms', 'N/A')} ms/img")
        print(f"    Params:   {metrics.get('model_params', 'N/A')}")
        print(f"    GFLOPs:   {metrics.get('model_gflops', 'N/A')}")
        print(f"    Size:     {metrics.get('model_size_mb', 'N/A')} MB")
        print()

        return metrics
