# Training, Benchmarking, and Selective Retraining Pipeline for YOLOv5/YOLOv8

> **Internship Project** — One-month end-to-end framework for object detection model training, benchmarking, and efficient selective retraining using Fisher importance scores (TinyTrain).

## Project Overview

This project extends an existing classification model pipeline to handle object detection with YOLOv5 and YOLOv8. It covers three main modules:

1. **Training Pipeline** — Configurable training for all 10 YOLO variants (Nano through XLarge) across two datasets, with MLflow experiment tracking.
2. **Benchmarking System** — Comparative evaluation of model variants on mAP, precision, recall, F1-score, inference speed, and GPU memory usage.
3. **TinyTrain** — Selective retraining module that uses Fisher information scores to identify the least important convolutional filters, freeze them, and perform efficient fine-tuning with reduced computational cost.