# YOLO Benchmark — Training, Benchmarking & TinyTrain

> Internship project: an end-to-end, config-driven framework for training and benchmarking
> **YOLOv5 / YOLOv8** object detection models, with **TinyTrain** selective filter freezing
> for efficient retraining.

## Overview

The pipeline is driven by a single YAML config (`configs/detector_config.yaml`). You declare
which models to train, on which dataset, with which hyperparameters, and whether TinyTrain
selective freezing should be applied — the system handles the rest.

Three modules:

| Module | What it does |
|--------|--------------|
| **Training** | Trains any enabled YOLO model from the config, optionally with TinyTrain filter freezing |
| **Evaluation** | Loads trained weights and benchmarks mAP50, mAP50-95, precision, recall, F1, inference speed, params, FLOPs |
| **Visualization** | Generates comparison bar charts / plots from each experiment's `comparison.csv` |

### TinyTrain

TinyTrain (`training/tiny_train.py`) is a selective-retraining method:

1. Loads a pretrained model and estimates **Fisher information** per conv filter using a
   sample of the training set.
2. Ranks filters by a **multi-objective importance metric** (Fisher score normalized by
   params and MACs per layer).
3. Builds a nested freeze list — the least important filters (up to `freeze_portion`)
   stay frozen while the rest are fine-tuned.

When a run has `tiny_train.enabled: true` (from its own section or the global block), the
detector (`models/detection/yolo_detector.py`) computes this list and passes it as the
`freeze=` argument to the underlying trainer instead of the plain `freeze` count.

## Project layout

```
├── main.py                      # CLI entry point (train / evaluate / list runs)
├── configs/
│   ├── detector_config.yaml     # All runs, datasets, hyperparameters, global settings
│   └── config_loader.py         # Parses YAML into DetectorConfig / RunConfig objects
├── models/
│   ├── base_model.py            # Abstract BaseModel interface
│   ├── __init__.py              # create_model() factory
│   └── detection/yolo_detector.py  # YOLOv5/v8 wrapper (load, train, predict, val, TinyTrain)
├── training/
│   ├── base_trainer.py          # Abstract BaseTrainer
│   ├── tiny_train.py            # Fisher-importance selective filter freezing
│   ├── trainer_factory.py       # create_trainer() factory
│   └── detector/yolo_trainer.py # Concrete YOLO trainer (pre-flight checks + train)
├── evaluation/
│   ├── base_evaluation.py       # Abstract BaseEvaluator
│   ├── evaluator_factory.py     # create_evaluator() factory
│   ├── visualize.py             # generate_all_plots() — bar charts from comparison.csv
│   └── detector/yolo_evaluator.py  # Weights resolution, val, metrics, CSV/JSON output
├── pipeline_utils/
│   └── logging_utils.py         # setup_logger / get_logger
├── dataset/                     # YOLO-format datasets (see Datasets below)
├── outputs/detection/           # Experiment folders + comparison.csv + plots
└── download_dataset.py          # Pulls datasets from Roboflow via RF_API_KEY
```

The three factories (`create_model`, `create_trainer`, `create_evaluator`) share a strategy +
factory pattern: they inspect the run's `family` field and return the matching concrete class.

## Datasets

The datasets are **not included in the repo** — you must download them from Roboflow.
Each dataset is YOLO format: a folder with `data.yaml` + `train/`, `valid/`, `test/`
(images + label `.txt` files).

| Dataset | Folder | Classes | Roboflow workspace / project (version) |
|---------|--------|---------|----------------------------------------|
| Mechanical tools | `dataset/mechanical_tools/` | drill, hammer, pliers, screwdriver, wrench (5) | `mechanical-tools` / `mechanical-tools-10000` (v3) |
| SMD components | `dataset/smd_components/` | condensator, diode, resistor, transistor (4) | `dainius` / `smdcomponents` (v6) |
| Drillbit detection | `dataset/drillbit_detection/` | tool, tool_b, void (3) | `small-objects-detection` / `drillbit-detection` (v3) |

### Downloading

`download_dataset.py` fetches all three datasets (it skips any that already exist):

```bash
cp .env.example .env        # add your Roboflow API key
python download_dataset.py
```

Or download a specific one by running its function from the script
(e.g. `python -c "from download_dataset import download_mechanical_tools as f; from roboflow import Roboflow; f(Roboflow(api_key='...'))"`).

To download manually from [Roboflow Universe](https://universe.roboflow.com): open the
project above, pick the listed version, and export in **YOLOv8 format**. The downloaded
archive extracts into `dataset/<name>/` with the expected layout.

Each dataset's `data.yaml` uses an absolute `path:` plus relative `train/val/test` keys.
The `data_yaml`, `nc`, and `names` fields in a run's `dataset` section must match it.

## Configuration

`configs/detector_config.yaml` holds a list of **runs**. Each run declares:

- `model`: family (`yolov5` / `yolov8`), variant (`n`/`s`/`m`/`l`/`x`), task
- `dataset`: `data_yaml`, `nc`, `names`
- `hyperparameters`: learning rate, augmentation (hsv, fliplr, mosaic, mixup, …), loss gains
- `training`: epochs, batch, workers, device, seed, optimizer, pretrained, amp, resume, …
- `inference`: `imgsz`, `conf_thres`, `iou_thres`, `max_det`, …
- `tiny_train`: optional per-run override (`enabled`, `imp_samples`, `imp_batch_size`, `freeze_portion`)
- `enabled`: `true`/`false`

Top-level sections control global behavior:

```yaml
evaluation:   # split, batch, conf_thres, iou_thres, device
tiny_train:   # global defaults inherited by runs that don't override
```

Run the `--enabled` flag to see which runs are active and their TinyTrain settings.

## Usage

```bash
# To list all the arguments :
python main.py -h # or --help

# List enabled runs and their config
python main.py --enabled

# Train all enabled runs into an experiment folder
python main.py --train --name exp_001

# Train a single run, overriding epochs / device / freeze ratio
python main.py --train --run mech_yolov5_small_baseline --epochs 20 --device mps
python main.py --train --freeze-ratio 0.3 --name exp_001

# Evaluate all trained runs (needs --name to find the experiment)
python main.py --evaluate --name exp_001

# Train then evaluate in one go
python main.py --train --evaluate --name exp_001
```

Experiments land in `outputs/detection/<name>/`; the experiment name auto-increments
(`experiment_001`, …) when `--name` is omitted. Each evaluation run appends a row to the
experiment's `comparison.csv` and writes per-run `metrics.json`.

## Output

```
outputs/detection/<experiment>/
├── <run_id>/                    # per-run evaluation
│   ├── metrics.json
│   └── weights/best.pt
├── comparison.csv               # one row per run (mAP, speed, params, …)
├── plots/                       # generated bar charts
└── pipeline.log
```

## Setup

### 1. Clone the vendored engines

The pipeline runs against local clones of the YOLO codebases (not PyPI releases), so clone
them first. These are **forks** with the TinyTrain modifications applied on the `tinytrain`
branch:

```bash
git clone -b tinytrain https://github.com/AliElKateb/yolov5.git yolov5
git clone -b tinytrain https://github.com/AliElKateb/ultralytics.git ultralytics_repo
```

### 2. Create the environment

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

> `yolov5/` needs no install — it's added to `sys.path` at runtime by
> `models/detection/yolo_detector.py`.

### 3. Download the datasets

The datasets are not included in the repo. Download them from Roboflow (see
[Datasets](#datasets)):

```bash
cp .env.example .env                   # add RF_API_KEY
python download_dataset.py
```

## Credits

The object-detection engines are vendored in this repository. Credit to their authors:

| Project | Role | License |
|---------|------|---------|
| [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (`yolov5/`) | YOLOv5 training/validation engine | AGPL-3.0 |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`ultralytics_repo/`) | YOLOv8 training/validation engine | AGPL-3.0 |

**TinyTrain** (`training/tiny_train.py`) implements selective filter freezing based on Fisher
information scores, as described in the TinyTrain paper; the implementation here is original
to this project.

Datasets are provided by [Roboflow](https://roboflow.com) (see the `dataset/README.roboflow.txt`
files in each dataset folder for attribution).
