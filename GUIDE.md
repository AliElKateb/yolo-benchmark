# YOLO Benchmarking Pipeline — Complete Guide

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [The Config System](#3-the-config-system)
4. [The Model Layer](#4-the-model-layer)
5. [The Training Layer](#5-the-training-layer)
6. [The Evaluation Layer](#6-the-evaluation-layer)
7. [The Entry Point (main.py)](#7-the-entry-point-mainpy)
8. [How Everything Fits Together](#8-how-everything-fits-together)
9. [Usage Examples](#9-usage-examples)
10. [Dataset Format](#10-dataset-format)
11. [Output Structure](#11-output-structure)

---

## 1. Project Overview

This project trains, evaluates, and compares **YOLOv5** and **YOLOv8** object detection models across all size variants (Nano, Small, Medium, Large, XLarge) — 10 models total.

It has three main modules:

| Module | Purpose |
|--------|---------|
| **Training** | Train any combination of YOLO models on your dataset |
| **Evaluation** | Benchmark trained models on mAP, precision, recall, F1, speed, etc. |
| **Config** | YAML-driven configuration that controls everything |

The pipeline is driven entirely by a single YAML config file. You define which models to run, with which hyperparameters, on which dataset, and the system handles the rest.

---

## 2. Project Structure

```
ubo_project/
├── main.py                          # Entry point (train & evaluate)
├── configs/
│   ├── detector_config.yaml         # All model configs & dataset settings
│   └── config_loader.py             # Parses the YAML into Python objects
├── models/
│   ├── base_model.py                # Abstract base class for all models
│   ├── __init__.py                  # create_model() factory
│   └── detection/
│       ├── __init__.py              # Exports YOLODetector
│       └── yolo_detector.py         # YOLOv5/v8 wrapper (load, train, predict, val)
├── training/
│   ├── base_trainer.py              # Abstract base trainer
│   ├── trainer_factory.py           # create_trainer() factory
│   ├── __init__.py
│   └── detector/
│       ├── __init__.py
│       └── yolo_trainer.py          # Concrete trainer for YOLO
├── evaluation/
│   ├── base_evaluation.py           # Abstract base evaluator
│   ├── evaluator_factory.py         # create_evaluator() factory
│   ├── __init__.py
│   └── detector/
│       ├── __init__.py
│       └── yolo_evaluator.py        # Concrete evaluator for YOLO
├── dataset/                         # Your datasets go here
│   └── cat_dog.v1i.yolov8/         # Example: cat vs dog dataset (YOLO format)
├── runs/                            # Auto-created by Ultralytics during training
│   └── detect/
│       └── yolov8_nano/            # One folder per trained model
│           └── weights/
│               ├── best.pt         # Best checkpoint (highest mAP)
│               └── last.pt         # Final epoch checkpoint
└── outputs/
    └── detection_experiments/       # Evaluation results stored here
        └── experiment_20260703_204343/
            ├── yolov8_nano/
            │   └── metrics.json
            └── comparison.csv
```

---

## 3. The Config System

### 3.1 The YAML file (`configs/detector_config.yaml`)

Everything starts here. The config defines **runs** — each run is one model to train and evaluate. Each run has:

```yaml
runs:
  - run_id: "yolov8_nano"          # Unique identifier for this run
    model:
      family: "yolov8"             # "yolov5" or "yolov8"
      variant: "n"                 # n, s, m, l, or x
      task: "detect"               # Always "detect" for now
    dataset:
      data_yaml: "./dataset/cat_dog.v1i.yolov8/data.yaml"  # Path to dataset
      nc: 2                        # Number of classes
      names: ["Cat", "Dog"]        # Class names
    hyperparameters:
      lr: 0.01                     # Learning rate
      momentum: 0.937              # SGD momentum
      weight_decay: 0.0005         # L2 regularization
      box: 7.5                     # Box loss gain (v8 style)
      cls: 0.5                     # Class loss gain
      dfl: 1.5                     # Distribution focal loss (v8)
      # ... augmentation params: hsv_h, flipud, mosaic, mixup, etc.
    training:
      epochs: 100                  # How many epochs to train
      batch: 16                    # Batch size
      device: "cpu"                # "cpu", "cuda:0", "mps", etc.
      optimizer: "AdamW"           # "SGD" for v5, "AdamW" for v8
      seed: 42                     # Random seed
      project: "runs"              # Output directory root
      name: "yolov8_nano"          # Output subdirectory
      # ... patience, resume, amp, cache, etc.
    inference:
      imgsz: [640, 640]            # Input image size
      conf_thres: 0.25             # Confidence threshold
      iou_thres: 0.45              # IoU threshold for NMS
      max_det: 300                 # Maximum detections per image
    metadata:
      description: "YOLOv8 Nano — fastest variant"
      tags: ["yolov8", "nano", "baseline"]
      notes: ""
    enabled: true                  # false = skip this run
```

There are also **top-level** settings:

```yaml
evaluation:
  output_dir: "./outputs/detection_experiments"
  split: "test"                    # "val" or "test"
  batch: 16
  conf_thres: 0.25
  iou_thres: 0.45
  half: false
  device: "cpu"
```

And a **commented-out TinyTrain** section at the bottom (for future use).

The `enabled: true/false` flag controls which runs are active. Currently, only 4 runs are enabled:
- `yolov5_nano`
- `yolov5_small`
- `yolov8_nano`
- `yolov8_small`

The medium/large/xlarge variants are disabled (set to `enabled: false`) since they're heavier.

### 3.2 The Config Loader (`configs/config_loader.py`)

The YAML file is raw text. To use it in Python code, we need to parse it. That's what `config_loader.py` does.

**`DetectorConfig`** class:
- Loads the YAML file with `yaml.safe_load()`
- Creates a `RunConfig` object for each run in the list
- Provides methods to filter runs:
  - `.runs` — all runs
  - `.enabled_runs` — only those with `enabled: true`
  - `.get_run("yolov8_nano")` — find by run_id
  - `.runs_by_family("yolov5")` — all v5 runs

**`RunConfig`** class:
- Wraps a single run's dict
- Exposes sections as properties:
  - `.run_id`, `.enabled`, `.family`, `.variant`
  - `.model`, `.dataset`, `.hyperparameters`, `.training`, `.inference`, `.metadata`
- `.global_config` — access top-level settings (like `evaluation`) from any run

**`load_config()`** — convenience function:
```python
from configs.config_loader import load_config
cfg = load_config()                          # loads the default YAML
run = cfg.get_run("yolov8_nano")
print(run.hyperparameters["lr"])             # 0.01
print(run.training["epochs"])                # 100
```

---

## 4. The Model Layer

The model layer wraps Ultralytics' YOLO and provides a consistent interface for all operations.

### 4.1 Base Model (`models/base_model.py`)

Abstract class that defines what every model must be able to do:

| Method | Purpose |
|--------|---------|
| `load(weights_path)` | Load pretrained `.pt` weights |
| `train(**kwargs)` | Run training |
| `predict(source, **kwargs)` | Run inference on images |
| `val(**kwargs)` | Run validation to get metrics |
| `save(path)` | Save model weights |
| `export(format)` | Export to ONNX/TorchScript |

Properties: `run_id`, `family`, `variant`, `config`

### 4.2 YOLO Detector (`models/detection/yolo_detector.py`)

The concrete implementation that wraps Ultralytics' `YOLO` class. This is where the config values actually get mapped to Ultralytics function calls.

**`load()`**:
```python
model.load()           # Auto-resolves to "yolov8n.pt"
model.load("custom.pt")  # Load custom weights
```

**`train()`**:
- Reads `hyperparameters`, `training`, `dataset`, `inference` from the RunConfig
- Maps every config field to the corresponding Ultralytics argument name (e.g., `lr` → `lr0`, `conf_thres` → `conf`)
- Calls `model.train(**args)` with all mapped parameters
- Any `**kwargs` overrides the config values

**`predict()`**:
- Reads inference settings from config (imgsz, conf, iou, max_det, etc.)
- Calls `model.predict()` with those settings
- Returns Ultralytics Results objects (boxes, confidence, class IDs)

**`val()`**:
- Reads dataset `data_yaml` and the global `evaluation` section
- Calls `model.val()` which runs the full validation pipeline
- Returns a `DetMetrics` object with mAP, precision, recall, etc.

### 4.3 Model Factory (`models/__init__.py`)

The `create_model()` function takes a `RunConfig` and returns the right model:

```python
def create_model(config: RunConfig) -> BaseModel:
    if config.family in ("yolov5", "yolov8"):
        return YOLODetector(config)
    raise ValueError(f"Unknown model family: {config.family}")
```

This is how the system **detects which model to run based on the run_id**. You pass a RunConfig, and the factory checks the `family` field to instantiate the correct class.

---

## 5. The Training Layer

The training layer adds orchestration on top of the model — pre-flight checks, logging, and result tracking.

### 5.1 Base Trainer (`training/base_trainer.py`)

Abstract class with:
- `train()` — the main method (abstract)
- `validate_dataset()` — checks that the `data_yaml` file exists before training

### 5.2 YOLO Trainer (`training/detector/yolo_trainer.py`)

The concrete trainer that runs the actual training:

```python
class YOLOTrainer(BaseTrainer):
    def train(self, **kwargs):
        # 1. Validate dataset exists
        # 2. Load pretrained weights
        # 3. Call model.train() with config parameters
        # 4. Return results
```

It prints a clear header so you know what's running:
```
============================================================
  Starting training: yolov8_nano
  Model: yolov8 n
============================================================
```

### 5.3 Trainer Factory (`training/trainer_factory.py`)

Maps model families to their trainers. Same pattern as the model factory.

---

## 6. The Evaluation Layer

This is where benchmarking happens. After a model has been trained, the evaluator loads its saved weights and runs validation to get all metrics.

### 6.1 Base Evaluator (`evaluation/base_evaluation.py`)

Abstract class with `evaluate()` method that returns a metrics dict.

### 6.2 YOLO Evaluator (`evaluation/detector/yolo_evaluator.py`)

This is the most complex component. It does the following:

**Step 1: Find trained weights (`_resolve_weights`)**
- Checks multiple path patterns where Ultralytics might have saved the weights
- Tries `best.pt` first, falls back to `last.pt`
- Raises a clear error if no weights are found

**Step 2: Set up output directory (`_setup_output_dir`)**
- Creates a timestamped folder: `outputs/detection_experiments/experiment_20260703_204343/`
- Within it, creates a subfolder for this run: `.../yolov8_nano/`

**Step 3: Extract metrics (`_extract_metrics`)**
From Ultralytics' validation results, it extracts:

| Metric | Source | Description |
|--------|--------|-------------|
| `mAP50` | `box.map50` | Mean AP at IoU=0.50 |
| `mAP50-95` | `box.map` | Mean AP averaged over IoU 0.50–0.95 |
| `Precision` | `box.mp` | Mean precision across classes |
| `Recall` | `box.mr` | Mean recall across classes |
| `F1-Score` | Calculated | Harmonic mean of P and R |
| `Inference speed` | `speed` dict | Preprocess/inference/postprocess ms |
| `Model params` | `model.parameters()` | Total trainable parameters |
| `Model GFLOPs` | Calculated | Giga-FLOPs (parameters × 2 / 1e9) |
| `Model size` | File size | Weights file size in MB |

Per-class breakdown (Cat, Dog) is also saved.

**Step 4: Save results (`_save_results`)**
- Saves all metrics as `metrics.json` in the run's subfolder
- Appends a row to `comparison.csv` in the experiment folder

**Step 5: Print summary**
```
  Metrics for yolov8_nano:
    mAP50:    92.5
    mAP50-95: 65.3
    P:        88.2
    R:        90.1
    F1:       89.1
    Speed:    119.08 ms/img
    Params:   3006038
    GFLOPs:   0.01
    Size:     5.96 MB
```

### 6.3 Evaluator Factory (`evaluation/evaluator_factory.py`)

Same factory pattern — maps family to evaluator.

---

## 7. The Entry Point (`main.py`)

`main.py` ties everything together.

### 7.1 Command-line arguments

| Argument | Effect |
|----------|--------|
| `--train` | Train all enabled models |
| `--evaluate` | Evaluate all trained models |
| `--run yolov8_nano` | Only process this specific run |
| `--epochs 50` | Override epochs for all runs |

If you specify both `--train` and `--evaluate`, it does training first, then evaluation.

### 7.2 How `run_training()` works

```
For each enabled run:
  1. Create the model:          create_model(run)     → YOLODetector
  2. Create the trainer:        create_trainer(model) → YOLOTrainer
  3. Train:                     trainer.train()       → loads weights, runs training
  4. Store results
  5. Print summary
```

### 7.3 How `run_evaluation()` works

```
For each enabled run:
  1. Create the model:          create_model(run)     → YOLODetector
  2. Create the evaluator:      create_evaluator(model) → YOLOEvaluator
  3. Evaluate:                  evaluator.evaluate()  → finds weights, runs val, saves metrics
  4. Store metrics
  5. Print summary
```

---

## 8. How Everything Fits Together

Here's the full flow:

```
main.py
  │
  ├── load_config()                    # Reads detector_config.yaml
  │   └── DetectorConfig
  │       └── RunConfig per run
  │
  ├── create_model(run_config)          # Factory: checks family
  │   └── YOLODetector(run_config)     # Wraps Ultralytics YOLO
  │
  ├── create_trainer(model, config)     # Factory: checks family
  │   └── YOLOTrainer(model, config)
  │       ├── validate_dataset()        # Does data.yaml exist?
  │       ├── model.load()              # Download/load yolov8n.pt
  │       └── model.train(              # Map config → Ultralytics args
  │               data=data_yaml,       #   Dataset
  │               epochs=100,           #   Training params
  │               lr0=0.01,             #   Hyperparameters
  │               hsv_h=0.015,          #   Augmentation
  │               ...
  │           )
  │
  └── create_evaluator(model, config)   # Factory: checks family
      └── YOLOEvaluator(model, config)
          ├── _resolve_weights()        # Find best.pt in runs/detect/...
          ├── model.load(weights)       # Load trained weights
          ├── model.val()               # Run validation on test split
          ├── _extract_metrics()        # Get mAP, P, R, speed, etc.
          ├── _save_results()           # Write metrics.json + comparison.csv
          └── print summary
```

### Design Patterns Used

**Factory Pattern** — `create_model()`, `create_trainer()`, `create_evaluator()` all use the same approach: check the `family` field and return the right concrete class. Adding a new model family (e.g., YOLOv11) means:
1. Create the new model/trainer/evaluator classes
2. Add the family name to the factory functions

**Strategy Pattern** — `BaseModel` / `BaseTrainer` / `BaseEvaluator` define the interface. YOLO-specific implementations handle the details.

**Config-driven** — The entire pipeline is driven by `detector_config.yaml`. Changing hyperparameters, datasets, or model variants requires editing the YAML file, not the Python code.

---

## 9. Usage Examples

```bash
# Train all 4 enabled models (nano + small for v5 and v8)
python main.py --train

# Train just one model for a quick test
python main.py --train --run yolov8_nano --epochs 2

# Evaluate all trained models
python main.py --evaluate

# Evaluate a single model
python main.py --evaluate --run yolov8_nano

# Full pipeline: train for 50 epochs then evaluate everything
python main.py --train --epochs 50 --evaluate

# Compare different hyperparameters:
# Run 1
python main.py --train --epochs 50 --evaluate
# Edit config, run again
python main.py --train --epochs 100 --evaluate
# Results go to separate experiment_* folders for comparison
```

---

## 10. Dataset Format

The dataset must be in **YOLO format** with:

```
dataset/
└── your_dataset_name/
    ├── data.yaml              # Dataset config
    ├── train/
    │   ├── images/            # Training images
    │   └── labels/            # YOLO .txt labels (class_id cx cy w h)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

**`data.yaml`** content:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['Cat', 'Dog']
```

Paths in `data.yaml` are relative to the yaml file's location.

You can export datasets from Roboflow in YOLOv8 format — it creates this exact structure.

To use a different dataset, update the `data_yaml` field in `detector_config.yaml`:
```yaml
dataset:
  data_yaml: "./dataset/your_dataset/data.yaml"
  nc: 5                          # Update to your number of classes
  names: ["class1", "class2"]    # Update to your class names
```

---

## 11. Output Structure

### Training output (`runs/`)

Created automatically by Ultralytics during training:

```
runs/detect/yolov8_nano/
├── weights/
│   ├── best.pt              # Best checkpoint (by validation mAP)
│   └── last.pt              # Last epoch checkpoint
├── labels.jpg               # Training labels visualization
├── confusion_matrix.png
├── results.csv              # Per-epoch metrics
├── results.png              # Training curves plot (loss, mAP, etc.)
├── train_batch*.jpg         # Sample batch visualizations
└── val_batch*.jpg           # Validation predictions
```

### Evaluation output (`outputs/`)

```
outputs/detection_experiments/
└── experiment_20260703_204343/          # Timestamp = unique experiment
    ├── yolov8_nano/
    │   └── metrics.json                 # All metrics for this run
    ├── yolov8_small/
    │   └── metrics.json
    └── comparison.csv                   # One row per run, easy to import in Excel
```

Every time you run evaluation, a new timestamped experiment folder is created. This lets you compare results across different hyperparameter settings:

```
outputs/detection_experiments/
├── experiment_20260703_100000/          # First experiment (50 epochs, lr=0.01)
│   ├── yolov8_nano/metrics.json
│   └── comparison.csv
└── experiment_20260703_110000/          # Second experiment (100 epochs, lr=0.001)
    ├── yolov8_nano/metrics.json
    └── comparison.csv
```

**`metrics.json`** content:
```json
{
  "map50_95": 65.3,
  "map50": 92.5,
  "precision": 88.2,
  "recall": 90.1,
  "per_class": [
    { "class": "Cat", "map50_95": 62.1, "map50": 90.0 },
    { "class": "Dog", "map50_95": 68.5, "map50": 95.0 }
  ],
  "f1_score": 89.1,
  "preprocess_ms": 2.1,
  "inference_ms": 119.1,
  "postprocess_ms": 2.1,
  "total_per_image_ms": 123.3,
  "model_params": 3006038,
  "model_gflops": 0.01,
  "model_size_mb": 5.96,
  "run_id": "yolov8_nano",
  "family": "yolov8",
  "variant": "n",
  "weights_path": "runs/detect/yolov8_nano/weights/best.pt",
  "dataset": "./dataset/cat_dog.v1i.yolov8/data.yaml"
}
```

**`comparison.csv`** — Aggregated table of all runs in an experiment for easy comparison in Excel or pandas.
