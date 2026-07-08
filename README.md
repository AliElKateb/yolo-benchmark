# Training, Benchmarking, and Selective Retraining Pipeline for YOLOv5/YOLOv8

> **Internship Project** — One-month end-to-end framework for object detection model training, benchmarking, and efficient selective retraining using Fisher importance scores (TinyTrain).

## Project Overview

This project extends an existing classification model pipeline to handle object detection with YOLOv5 and YOLOv8. It covers three main modules:

1. **Training Pipeline** — Configurable training for all 10 YOLO variants (Nano through XLarge) across two datasets, with MLflow experiment tracking.
2. **Benchmarking System** — Comparative evaluation of model variants on mAP, precision, recall, F1-score, inference speed, and GPU memory usage.
3. **TinyTrain** — Selective retraining module that uses Fisher information scores to identify the least important convolutional filters, freeze them, and perform efficient fine-tuning with reduced computational cost.

## Getting Started

### Prerequisites

- Python 3.9+
- [Roboflow](https://roboflow.com) account (free tier)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/AliElKateb/yolo-benchmark.git
cd yolo-benchmark

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
# Option A — pip (recommended for most users)
pip install -r requirements.txt

# Option B — uv (faster alternative)
# pip install uv && uv sync
```

### Dataset Setup

1. Get a free API key from [Roboflow](https://roboflow.com).
2. Create a `.env` file in the project root:
   ```
   RF_API_KEY=your_roboflow_api_key
   ```
3. Download the datasets:
   ```
   python download_dataset.py
   ```

### Datasets

| Dataset | Images | Classes | Source |
|---------|--------|---------|--------|
| **SMD Components** | 7.8k | Condensator, Diode, Resistor, Transistor | [Roboflow](https://universe.roboflow.com/dainius/smdcomponents) |
| **Mechanical tools-10000** | 9.3k | drill, hammer, pliers, screwdriver, wrench | [Roboflow](https://universe.roboflow.com/mechanical-tools/mechanical-tools-10000) |