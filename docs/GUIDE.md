# YOLO Benchmarking Pipeline — Complete Guide

This guide is the starting point for any developer working with this repository.
It explains what the pipeline does, how the code is organized, how to set up a
working environment, and how to run each stage. It mirrors the current state of
the code (two-stage baseline/retrain workflow with TinyTrain).

The shorter [README](../README.md) gives a quick overview; this document goes
into the detail a contributor needs. Metric/ranking explanations live in
[docs/RANKING.md](RANKING.md), and planned work is tracked in
[UPCOMING.md](../UPCOMING.md).

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What a New Developer Needs to Know](#2-what-a-new-developer-needs-to-know)
3. [Setup](#3-setup)
4. [Project Structure](#4-project-structure)
5. [The Config System](#5-the-config-system)
6. [The Model Layer](#6-the-model-layer)
7. [The Training Layer](#7-the-training-layer)
8. [The Evaluation Layer](#8-the-evaluation-layer)
9. [The Merge & Ranking Stage](#9-the-merge--ranking-stage)
10. [The Entry Point (main.py)](#10-the-entry-point-mainpy)
11. [How Everything Fits Together](#11-how-everything-fits-together)
12. [Worked Example — First Run](#12-worked-example--first-run)
13. [Dataset Format](#13-dataset-format)
14. [Output Structure](#14-output-structure)

---

## 1. Project Overview

This project trains, evaluates, and compares **YOLOv5** and **YOLOv8** object
detection models using a **two-stage baseline/retrain workflow**:

| Stage | What happens |
|-------|--------------|
| **Baseline (`--train`)** | Train each enabled model from COCO-pretrained weights on the **70%** baseline split |
| **Retrain (`--retrain`)** | Continue from the baseline checkpoint on the **30%** retrain split, either with **TinyTrain** (selective filter freezing) or the whole model |
| **Evaluate** | Benchmark every model on the **shared `valid/`** split (mAP50, mAP50-95, precision, recall, F1, speed, params, size) |
| **Merge (`--merge`)** | Combine all baseline + retrain results into one table and **rank** them by a composite score |
| **Visualize** | Generate bar/radar/dashboard charts from each experiment's `comparison.csv` |

It also implements **TinyTrain** (`training/tiny_train.py`), a selective
filter-freezing retraining method based on Fisher information: the least
important conv filters are frozen while the rest are fine-tuned, which can cut
retraining cost with little accuracy loss.

Everything is driven by a single YAML config (`configs/detector_config.yaml`).
You declare which runs to execute and how; the system handles the rest.

> **Important — this pipeline runs against vendored forks of the YOLO
> engines**, not the PyPI releases. The `yolov5/` and `ultralytics_repo/`
> directories carry the TinyTrain modifications and are **not** committed to
> this repo — you must clone them yourself (see [Setup](#3-setup)).

---

## 2. What a New Developer Needs to Know

- **Three factories, one pattern.** `create_model()`, `create_trainer()`, and
  `create_evaluator()` all inspect a run's `family` field (`"yolov5"` /
  `"yolov8"`) and return the matching concrete class. Adding a new family means
  adding classes + registering them in the three factories.
- **YOLOv5 vs YOLOv8 use different engines.** YOLOv8 goes through Ultralytics'
  `YOLO` class; YOLOv5 checkpoints can't be loaded by Ultralytics, so they are
  loaded/trained/validated through the **vendored `yolov5/` repo**, which is
  put on `sys.path` (see `models/detection/yolo_detector.py`).
- **Two-list dataset layout.** Each run's `dataset` section carries both
  `baseline_data` and `retrain_data` (plus the original `data_yaml`). The
  splits are produced by `split_dataset.py` and share one `valid/` set.
- **`--retrain` and `--merge` are strict.** They require `--name`, and
  `tiny_train` also requires `--freeze-ratio`. Missing flags abort **before**
  anything runs (no silent defaults, no prompts).
- **Outputs are git-ignored** (`outputs/`, `dataset/`, `runs/`, the vendored
  repos, and `*.pt` weights). Don't commit experiment artifacts.

---

## 3. Setup

### 3.1 Clone the vendored engines

The pipeline runs against local forks with the TinyTrain modifications on their
`tinytrain` branch:

```bash
git clone -b tinytrain https://github.com/AliElKateb/yolov5.git yolov5
git clone -b tinytrain https://github.com/AliElKateb/ultralytics.git ultralytics_repo
```

### 3.2 Create the environment

```bash
# Option A — pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ultralytics_repo/         # editable install of the vendored ultralytics

# Option B — uv (faster)
uv sync
uv pip install -e ultralytics_repo/      # editable install of the vendored ultralytics
source .venv/bin/activate
```

> `yolov5/` needs no install — it is appended to `sys.path` at runtime by
> `models/detection/yolo_detector.py`.

### 3.3 Download the datasets

The datasets are **not included in the repo**; pull them from Roboflow:

```bash
cp .env.example .env                     # then add your RF_API_KEY
python download_dataset.py
```

This fetches `mechanical_tools`, `smd_components`, and `drillbit_detection`
into `dataset/`, skipping any that already exist.

### 3.4 Prepare the two-stage splits

For a dataset you want to run the full baseline/retrain pipeline on, create the
70/30 training split (test/valid are copied into both so the split stays
identical):

```bash
python split_dataset.py --name mechanical_tools --baseline 0.7 --seed 42
```

This generates `dataset/mechanical_tools_baseline/` and
`dataset/mechanical_tools_retrain/`, each with its own `data.yaml` pointing at
the shared `valid/`.

---

## 4. Project Structure

```
yolo-benchmark/
├── main.py                              # CLI entry point (train / retrain / evaluate / merge / enabled)
├── configs/
│   ├── detector_config.yaml             # All runs, datasets, hyperparameters, global blocks
│   └── config_loader.py                 # DetectorConfig / RunConfig objects
├── models/
│   ├── base_model.py                    # Abstract BaseModel interface
│   ├── __init__.py                      # create_model() factory
│   └── detection/yolo_detector.py       # YOLOv5/v8 wrapper (load, train, predict, val, TinyTrain)
├── training/
│   ├── base_trainer.py                  # Abstract BaseTrainer (+ validate_dataset)
│   ├── tiny_train.py                    # Fisher-importance selective filter freezing
│   ├── trainer_factory.py               # create_trainer() factory
│   └── detector/yolo_trainer.py         # Concrete YOLO trainer (pre-flight checks + train)
├── evaluation/
│   ├── base_evaluation.py               # Abstract BaseEvaluator
│   ├── evaluator_factory.py             # create_evaluator() factory
│   ├── visualize.py                     # generate_all_plots() — bar/radar/dashboard charts
│   └── detector/yolo_evaluator.py       # Weights resolution, val, metrics, JSON/CSV output
├── pipeline_utils/
│   └── logging_utils.py                 # setup_logger() / get_logger()
├── dataset/                             # YOLO-format datasets (%_baseline / %_retrain splits included)
├── outputs/detection/                   # Experiment folders + comparison.csv + plots (git-ignored)
├── download_dataset.py                  # Pulls datasets from Roboflow via RF_API_KEY
├── split_dataset.py                     # Splits a dataset's train set into baseline(70%)/retrain(30%)
└── yolov5/, ultralytics_repo/           # Vendored engines (clone, git-ignored)
```

---

## 5. The Config System

### 5.1 The YAML file (`configs/detector_config.yaml`)

Everything starts here. The config holds a list of **runs** — each run is one
model. A run has these sections:

```yaml
runs:
  - run_id: "mech_yolov5_small"        # Unique identifier
    model:
      family: "yolov5"                 # "yolov5" or "yolov8"
      variant: "s"                     # n, s, m, l, x
      task: "detect"
    dataset:
      data_yaml: "./dataset/mechanical_tools/data.yaml"      # original full set
      baseline_data: "./dataset/mechanical_tools_baseline/data.yaml"  # 70% split
      retrain_data:  "./dataset/mechanical_tools_retrain/data.yaml"   # 30% split
      nc: 5
      names: ["drill", "hammer", "pliers", "screwdriver", "wrench"]
    hyperparameters:
      lr: 0.01
      momentum: 0.937
      # ... loss gains (box/cls/obj), augmentation (hsv_*, mosaic, mixup, ...)
    training:
      epochs: 15
      batch: 4
      workers: 2
      device: "mps"
      optimizer: "SGD"                 # "SGD" typical for v5, "AdamW" for v8
      # ... early_stopping, resume, pretrained, amp, freeze, cache, ...
    inference:
      imgsz: [416, 416]
      conf_thres: 0.25
      iou_thres: 0.45
      max_det: 300
    metadata:
      description: "YOLOv5 Small on Mechanical tools (baseline/retrain two-stage)"
      tags: ["yolov5", "small", "mechanical", "twostage"]
    enabled: true                      # false = skip this run
```

**Top-level** sections control global behavior (inherited by all runs):

```yaml
evaluation:    # split: "val", batch, conf_thres, iou_thres, half, device
retrain:       # default epochs + usage note
tiny_train:    # global defaults: enabled, imp_samples, imp_batch_size, freeze_portion
```

Key points:

- `RunConfig.baseline_data` / `RunConfig.retrain_data` **fall back** to
  `data_yaml` when the split keys are absent (`configs/config_loader.py:85-90`),
  so single-stage runs keep working.
- `evaluation.split: "val"` — baseline **and** retrained models are measured on
  the exact same validation set.
- `tiny_train` is global (`enabled: true`, `freeze_portion: 0.5`); a run can
  override it with its own inline `tiny_train:` block.
- The heavy preconfigured runs (`yolov5_*`, `yolov8_*`, `mech_yolov*_nano`,
  etc.) are `enabled: false`. Currently only `mech_yolov5_small` is enabled.

### 5.2 The Config Loader (`configs/config_loader.py`)

Two classes parse and validate the YAML:

**`DetectorConfig`**
- Loads the file with `yaml.safe_load()` and validates required keys
  (`run_id`, `model.family`, `model.variant`) with clear errors.
- `.runs` — all runs; `.enabled_runs` — only `enabled: true`
- `.get_run("mech_yolov5_small")` — lookup by id; `.runs_by_family(...)`
- `.global_config` — the top-level sections (everything except `runs`)

**`RunConfig`**
- Wraps a single run; exposes `.run_id`, `.enabled`, `.family`, `.variant`,
  `.model`, `.dataset`, `.baseline_data`, `.retrain_data`,
  `.hyperparameters`, `.training`, `.inference`, `.metadata`
- `.get(key, default)` — raw access; `.global_config` — top-level settings

**`load_config()`** convenience (also exposed via `main.py`):

```python
from configs.config_loader import load_config
cfg = load_config()
run = cfg.get_run("mech_yolov5_small")
print(run.baseline_data)     # dataset/mechanical_tools_baseline/data.yaml
print(run.training["epochs"])
```

---

## 6. The Model Layer

Wraps the underlying YOLO engines behind one consistent interface.

### 6.1 Base Model (`models/base_model.py`)

Abstract `BaseModel` defines the contract every model implements:

| Method | Purpose |
|--------|---------|
| `load(weights_path=None)` | Load `.pt` weights (defaults to `{family}{variant}.pt`) |
| `train(**kwargs)` | Run training (kwargs override config) |
| `predict(source, **kwargs)` | Run inference |
| `val(**kwargs)` | Run validation, return metrics |
| `save(path)` | Save weights |
| `export(format="onnx")` | Export to ONNX/TorchScript |

Properties: `run_id`, `family`, `variant`, `config`.

### 6.2 YOLO Detector (`models/detection/yolo_detector.py`)

The concrete implementation. The non-obvious part is the **two-engine
dispatch**:

- **YOLOv8** — loads with `ultralytics.YOLO(path)` and calls its `.train()`,
  `.val()`, `.predict()`.
- **YOLOv5** — Ultralytics refuses YOLOv5 checkpoints (they pickle
  `models.yolo`), so everything routes through the **vendored `yolov5/` repo**:
  - `_yolov5_import_context()` temporarily moves `yolov5/` to the front of
    `sys.path`, imports `utils`/`val`/`train` from it, then restores the
    original repos — this also resolves the repo-root `models/` vs
    `yolov5/models/` name clash.
  - `_load_yolov5_model()` uses `yolov5`'s own `attempt_load`.

**Config → engine mapping.** `train()` reads the run's `hyperparameters`,
`training`, `dataset`, and `inference` sections and maps each field to the
correct argument name for the active engine (e.g. `lr` → `lr0`,
`batch` → `batch_size` for v5, `conf_thres` → `conf`). `**override_kwargs`
win over config values, and `data`/`weights`/`tiny_train_enabled`/
`freeze_ratio` override kwargs are popped before the engine call.

**TinyTrain hook.** When `tiny_train.enabled` is true (or a `freeze_ratio` is
passed), `train()` runs `TinyTrain.apply(...)` to build a nested freeze list
and passes it as the engine's `freeze=` argument instead of the plain count.
The computed list is available afterwards via `model.last_freeze_list`.

### 6.3 Model Factory (`models/__init__.py`)

```python
def create_model(config: RunConfig) -> BaseModel:
    if config.family in ("yolov5", "yolov8"):
        return YOLODetector(config)
    raise ValueError(f"Unknown model family: {config.family}")
```

### 6.4 `val()` return shape

YOLOv8 returns an Ultralytics `DetMetrics` object; YOLOv5's `val.run()` returns
`(results, maps, times)`, which `val()` normalizes into a dict
(`mp`, `mr`, `map50`, `map50_95`, `speed`, `maps`, `losses`).

---

## 7. The Training Layer

Adds orchestration on top of the model — pre-flight checks, headers, and
result collection.

### 7.1 Base Trainer (`training/base_trainer.py`)

Abstract class holding `model` and `config`. Required `train(**kwargs)` and a
concrete `validate_dataset(data_yaml=None)` that checks the `data.yaml` exists
(optionally the baseline/retrain split yaml).

### 7.2 YOLO Trainer (`training/detector/yolo_trainer.py`)

The `YOLOTrainer.train()` flow:

```python
# 1. Print header (run_id + family/variant)
# 2. Pop 'weights' (None for baseline → COCO weights)
# 3. validate_dataset(data)         → FileNotFoundError if missing
# 4. self._model.load(weights)
# 5. self._model.train(weights=weights, **kwargs)
```

It prints a clear header and, when retraining, which checkpoint it continues
from.

### 7.3 Trainer Factory (`training/trainer_factory.py`)

`create_trainer(model, config)` → `YOLOTrainer` for `yolov5`/`yolov8`.

### 7.4 TinyTrain (`training/tiny_train.py`)

Selective filter freezing for the retrain stage:

1. **Estimate importance.** Load a checkpoint (the baseline weights on retrain,
   or COCO weights otherwise) and, on a sample of `imp_samples` images, accumulate
   gradients and compute a per-filter **Fisher-score-like** metric
   `(weight × gradient)²`.
2. **Multi-objective ranking.** Each conv layer's score is normalized by its
   param and MAC counts (`_compute_multi_obj_metric`), so cheap layers aren't
   penalized.
3. **Build the freeze list.** The `freeze_portion` least-important layers/filters
   are selected and encoded as a **nested list**
   `[[layer_name_with_underscores, idx, ...], ...]` (the format both YOLO
   engines expect for per-filter freezing).

Settings come from the run's `tiny_train` block or the global one
(`enabled`, `imp_samples`, `imp_batch_size`, `freeze_portion`).

---

## 8. The Evaluation Layer

Loads a trained model and benchmarks it on the **shared `valid/` split**.

### 8.1 Base Evaluator (`evaluation/base_evaluation.py`)

Abstract class; `evaluate(weights_path=None, **kwargs) → dict`.

### 8.2 YOLO Evaluator (`evaluation/detector/yolo_evaluator.py`)

`evaluate()` runs these steps:

**Step 1 — Resolve weights (`_resolve_weights`).**
Searches `outputs/detection/<experiment>/<run_id>/weights/{best,last}.pt`,
then `runs/detect|train`, then the `run_id-*` glob fallback. Raises a clear
`FileNotFoundError` otherwise.

**Step 2 — Setup output dirs (`_setup_output_dir`).**
Creates `outputs/detection/<exp>/<run_id>/`.

**Step 3 — Validate & extract metrics (`_extract_metrics`).**
Handles both the YOLOv8 `DetMetrics` object and the YOLOv5 dict form:

| Metric | Source |
|--------|--------|
| `map50_95` | `box.map` (v8) / `map50_95` (v5), ×100 |
| `map50` | `box.map50` / `map50`, ×100 |
| `precision` | `box.mp` / `mp`, ×100 |
| `recall` | `box.mr` / `mr`, ×100 |
| `f1_score` | `2·P·R / (P + R)` |
| `preprocess_ms` / `inference_ms` / `postprocess_ms` / `total_per_image_ms` | `speed` |
| `per_class` | `box.ap` / `box.ap50` (v8) |
| `model_params` | `sum(p.numel() ...)` |

**Step 4 — Save results (`_save_results`).**
Copies training artifacts into the run folder, writes `metrics.json`, and
appends/replaces the run's row in the experiment `comparison.csv`.

**Step 5 — Print summary** (mAP50, mAP50-95, P/R/F1, speed, params, size).

### 8.3 Evaluator Factory / Visualization

- `create_evaluator(model, config, experiment_name=None)` — factory.
- `evaluation/visualize.py::generate_all_plots(csv_path, output_dir)` — builds
  `mAP_comparison.png`, `precision_recall_f1.png`, `inference_speed.png`,
  `model_size.png`, `summary_dashboard.png`, `radar_chart.png` into
  `outputs/detection/<exp>/plots/`. Called automatically after every evaluate.

---

## 9. The Merge & Ranking Stage

`run_merge()` in `main.py` (see [docs/RANKING.md](RANKING.md) for the full
explanation):

1. Reads the baseline `outputs/detection/<name>/comparison.csv`.
2. Finds every sibling retrain experiment
   `outputs/detection/<name>_retrain_*` (suffix rules in
   `_retrain_dir_prefix()` — `tt_0.50` → `tt0.50`, `whole`).
3. Tags each row's Run ID with a strategy prefix (`base_`, `tt0.50_`, `whole_`).
4. Merges into `<name>_merged/comparison.csv`.
5. Ranks rows by a **composite score** (weighted, normalized 0–100) and writes
   `<name>_merged/ranking.csv`; prints a console table with highlights
   (best overall / best accuracy / fastest).
6. Regenerates plots for the merged table.

Run ID tags in `ranking.csv` and plots map back to freeze ratios
(`tt0.50_*` → 50%, `whole_*` → 0%, `base_*` → none).

---

## 10. The Entry Point (`main.py`)

### 10.1 Command-line arguments

| Argument | Effect |
|----------|--------|
| `--train` | Train the **baseline** split (70%) for enabled runs, then auto-evaluate on shared `valid/` |
| `--retrain` | Retrain a baseline experiment on the **retrain** split (30%) with `--strategy` |
| `--evaluate` | Evaluate trained models (standalone, or used by train/retrain) |
| `--merge` | Merge baseline + retrain CSVs, rank, save `ranking.csv` + plots |
| `--enabled` | List enabled runs and exit (no training) |
| `--name` | Experiment folder name (auto-increments for train/eval; **required** for retrain/merge) |
| `--run` | Target one specific run_id |
| `--epochs` / `--device` | Override for all runs |
| `--strategy` | `tiny_train` or `whole` (required for `--retrain`) |
| `--freeze-ratio` | 0.0–1.0; required for `--strategy tiny_train` |
| `--tag` | Override the `base_` prefix used on baseline rows during merge |
| `--merge-name` | Output folder for merge (default `<name>_merged`) |

**Validation (`parse_args`)**: combining `--train` with `--retrain` errors; a
mode must be given; retrain insists on `--name` + `--strategy` (+
`--freeze-ratio` for tiny_train). All errors happen up front, before any
training.

### 10.2 Stage details

- `run_training()` — for each enabled (or `--run`-targeted) run:
  `create_model` → `create_trainer` → `trainer.train(data=baseline_data,
  tiny_train_enabled=False, project=outputs/detection/<exp>)`. Baseline runs
  never apply TinyTrain. Results are then **auto-evaluated** and plotted.
- `run_retraining()` — requires `--name`; resolves the baseline checkpoint via
  `_find_baseline_weights()`, names the experiment
  `<base>_retrain_tt_0.50` or `<base>_retrain_whole`, trains with
  `data=retrain_data`, `weights=<baseline.pt>`, and
  `tiny_train_enabled=(strategy == "tiny_train")`, then auto-evaluates and
  writes `argument.yaml` (strategy, freeze_ratio, freeze list, baseline ref).
- `run_merge()` — see [section 9](#9-the-merge--ranking-stage).

### 10.3 Device detection

`detect_device()` picks `cuda:0` → `mps` → `cpu`, and normalizes
`cuda`/`gpu` → `0`. MPS fallbacks are forced via env vars at import time.

---

## 11. How Everything Fits Together

```
main.py
  ├── load_config()                        # yaml -> DetectorConfig -> RunConfig per run
  │       └── RunConfig.baseline_data / .retrain_data
  │
  ├── __TRAIN__ (baseline, 70% split)
  │   ├── create_model(run)                # factory -> YOLODetector
  │   ├── create_trainer(model, run)       # factory -> YOLOTrainer
  │   │     └── trainer.train(data=baseline_data, tiny_train_enabled=False)
  │   │           -> model.load()          # {family}{variant}.pt (COCO)
  │   │           -> model.train(...)      # config mapped to engine args
  │   └── run_evaluation(...) -> outputs/detection/<exp>/comparison.csv
  │   └── generate_all_plots(...)
  │
  ├── __RETRAIN__ (30% split, from baseline checkpoint)
  │   ├── _find_baseline_weights(<exp>)
  │   ├── trainer.train(data=retrain_data, weights=<baseline.pt>,
  │   │                tiny_train_enabled=True, freeze_ratio=X)
  │   │     -> TinyTrain.apply(...) -> nested freeze list -> engine freeze=
  │   │     -> outputs/detection/<exp>_retrain_tt_0.50/
  │   ├── _write_retrain_args()            # argument.yaml
  │   └── run_evaluation + plots
  │
  └── __MERGE__
      ├── read <base>/comparison.csv + all <base>_retrain_*/comparison.csv
      ├── tag run ids (base_, tt0.50_, whole_)
      ├── _rank_and_report()               # composite score -> ranking.csv
      └── generate_all_plots(merged)
```

### Design patterns

- **Factory** — one `create_*` per layer; dispatch on `family`.
- **Strategy** — `BaseModel` / `BaseTrainer` / `BaseEvaluator` define the
  interface; YOLO-specific implementations fill in the details.
- **Config-driven** — hyperparameters, datasets, splits, and enabled runs all
  come from `detector_config.yaml`; no code changes needed between experiments.

---

## 12. Worked Example — First Run

```bash
# 0. One-time setup
git clone -b tinytrain https://github.com/AliElKateb/yolov5.git yolov5
git clone -b tinytrain https://github.com/AliElKateb/ultralytics.git ultralytics_repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ultralytics_repo/
cp .env.example .env                      # add RF_API_KEY
python download_dataset.py                # pull datasets
python split_dataset.py --name mechanical_tools --baseline 0.7

# 1. See what's active
python main.py --enabled

# 2. Train the baseline and auto-evaluate (70% split)
python main.py --train --name exp_mech_yolov5s --epochs 15

# 3. Retrain from that baseline with TinyTrain at 50% and 90% freeze
python main.py --retrain --name exp_mech_yolov5s --strategy tiny_train --freeze-ratio 0.5
python main.py --retrain --name exp_mech_yolov5s --strategy tiny_train --freeze-ratio 0.9

# 4. Retrain the whole model (no freezing) as another comparison point
python main.py --retrain --name exp_mech_yolov5s --strategy whole

# 5. Merge everything and rank
python main.py --merge --name exp_mech_yolov5s

# Quick checks on a small subset
python main.py --train --run mech_yolov5_small --epochs 2 --device mps --name smoke_test
```

The merged table/ranking (with `rank_`, `Score`, `Delta vs Baseline`) lands in
`outputs/detection/exp_mech_yolov5s_merged/`.

---

## 13. Dataset Format

Datasets are **YOLO format**, downloaded from Roboflow:

```
dataset/your_dataset/
├── data.yaml              # dataset config
├── README.roboflow.txt    # attribution (generated by Roboflow)
├── train/
│   ├── images/
│   └── labels/            # YOLO .txt: class_id cx cy w h
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`data.yaml` (Roboflow style) uses an absolute `path:` plus relative keys:

```yaml
path: /abs/path/to/dataset/your_dataset
train: train/images
val: valid/images
test: test/images
nc: 5
names: ["drill", "hammer", "pliers", "screwdriver", "wrench"]
```

**Two-stage split.** For the baseline/retrain pipeline, run
`split_dataset.py` — it splits **only `train/`** into a 70% baseline and a 30%
retrain set, stratified by class signature, and copies `test/` + `valid/`
(unchanged) into both new folders so they share the identical eval set:

```
dataset/mechanical_tools_baseline/   # train (70%) + shared test + shared valid
dataset/mechanical_tools_retrain/    # train (30%) + shared test + shared valid
```

Point `baseline_data` / `retrain_data` at the generated `data.yaml` files in
each run's `dataset` section. A `names` / `nc` mismatch between the config and
`data.yaml` will break training — keep them consistent.

---

## 14. Output Structure

All artifacts live under `outputs/detection/` (git-ignored):

```
outputs/detection/
├── exp_mech_yolov5s/                          # baseline experiment
│   ├── mech_yolov5_small/                     # per-run evaluation
│   │   ├── metrics.json                       # all metrics for this run
│   │   ├── weights/best.pt                    # copied trained weights
│   │   └── ...                                # copied training artifacts
│   ├── comparison.csv                         # one row per run
│   ├── plots/                                 # auto-generated charts
│   └── pipeline.log
├── exp_mech_yolov5s_retrain_tt_0.50/          # TinyTrain 50% retrain
│   ├── argument.yaml                          # strategy, freeze_ratio, freeze list
│   └── comparison.csv
├── exp_mech_yolov5s_retrain_tt_0.90/
├── exp_mech_yolov5s_retrain_whole/
└── exp_mech_yolov5s_merged/                   # --merge output
    ├── comparison.csv                         # tagged rows (base_, tt0.50_, ...)
    ├── ranking.csv                            # ranked by composite score
    └── plots/
```

**`metrics.json`** example (v8):

```json
{
  "map50_95": 65.3,
  "map50": 92.5,
  "precision": 88.2,
  "recall": 90.1,
  "f1_score": 89.1,
  "preprocess_ms": 2.1,
  "inference_ms": 119.1,
  "postprocess_ms": 2.1,
  "total_per_image_ms": 123.3,
  "model_params": 3006038,
  "model_size_mb": 5.96,
  "run_id": "mech_yolov8_small",
  "family": "yolov8",
  "variant": "s",
  "weights_path": "outputs/detection/exp_mech_yolov5s/mech_yolov8_small/weights/best.pt",
  "dataset": "./dataset/mechanical_tools/data.yaml"
}
```

**`comparison.csv`** — aggregated table (one row per run) importable in Excel
or pandas, with columns: `Run ID, Family, Variant, mAP50, mAP50-95, Precision,
Recall, F1-Score, Inference (ms), Total (ms/img), Parameters, Size (MB),
Weights`.

`argument.yaml` (retrain experiments) records exactly what a retrain did —
strategy, freeze ratio, the nested freeze list, and the baseline experiment it
continued from — so any artifact on disk is reproducible.