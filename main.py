"""
Entry point for the YOLO benchmarking pipeline.

Supports training and evaluation modes. In training mode, it loads
the config and trains all enabled runs. In evaluation mode, it loads
trained weights and benchmarks each model, saving results to
outputs/detection/.

Usage:
    python main.py --train --name exp_001                          # train all enabled runs
    python main.py --train --run yolov8_nano --name exp_001        # train a single run
    python main.py --train --epochs 10 --device mps --name exp_001 # override epochs + device
    python main.py --train --freeze-ratio 0.3 --name exp_001       # override TinyTrain freeze ratio
    python main.py --evaluate --name exp_001                       # evaluate all trained runs
    python main.py --evaluate --run yolov8_nano --name exp_001     # evaluate a single run
    python main.py --train --evaluate --name exp_001               # train then evaluate
    python main.py --enabled                                        # list enabled runs and exit
"""

import os
import re
import argparse
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from configs.config_loader import load_config
from models import create_model
from training import create_trainer
from evaluation import create_evaluator
from evaluation.visualize import generate_all_plots
from pipeline_utils.logging_utils import setup_logger

log = setup_logger("pipeline", log_file="outputs/detection/pipeline.log")


def detect_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_device(device: str) -> str:
    dev = device.strip().lower()
    if dev in ("cuda", "gpu"):
        return "0"
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO benchmarking pipeline")

    parser.add_argument("--train", action="store_true", default=False, help="Run training for enabled models")
    parser.add_argument("--evaluate", action="store_true", default=False, help="Run evaluation on trained models")
    parser.add_argument("--run", type=str, default=None, help="Target a specific run_id (e.g. 'yolov8_nano')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs for all runs")
    parser.add_argument("--device", type=str, default=None, help="Override device for all runs (e.g. 'cpu', 'mps', 'cuda:0')")
    parser.add_argument("--freeze-ratio", type=float, default=None, help="Override TinyTrain freeze_portion (0.0–1.0) for all runs")
    parser.add_argument("--enabled", action="store_true", default=False, help="List enabled runs and exit")
    parser.add_argument("--name", type=str, default=None, help="Experiment folder name (auto-generated if omitted)")

    args = parser.parse_args()
    if args.enabled:
        return args
    if not args.train and not args.evaluate:
        parser.print_help()
        print("\n  Error: specify at least one of --train or --evaluate.")
        sys.exit(1)

    return args


def resolve_experiment_name(name: str | None) -> str:
    if name:
        return name
    base = Path("outputs/detection")
    base.mkdir(parents=True, exist_ok=True)
    max_num = 0
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("experiment_"):
            m = re.fullmatch(r"experiment_(\d+)", d.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return f"experiment_{max_num + 1:03d}"


def run_training(cfg, args, exp_name: str) -> dict:
    runs = _resolve_runs(cfg, args)
    log.info("Training %d run(s) in experiment '%s'", len(runs), exp_name)

    results = {}
    for run in runs:
        try:
            model = create_model(run)
            trainer = create_trainer(model, run)

            override_kwargs = {
                "project": f"outputs/detection/{exp_name}",
                "name": run.run_id,
                "exist_ok": True,
            }
            if args.epochs is not None:
                override_kwargs["epochs"] = args.epochs
            if args.device is not None:
                override_kwargs["device"] = normalize_device(args.device)
            if args.freeze_ratio is not None:
                override_kwargs["freeze_ratio"] = args.freeze_ratio

            log.info("Training %s (family=%s, variant=%s)",
                     run.run_id, run.family, run.variant)
            result = trainer.train(**override_kwargs)
            results[run.run_id] = result
            log.info("Training complete: %s", run.run_id)

        except Exception as e:
            log.error("Training %s failed: %s", run.run_id, e, exc_info=True)
            results[run.run_id] = {"error": str(e)}

    return results


def run_evaluation(cfg, args, exp_name: str) -> dict:
    runs = _resolve_runs(cfg, args)
    log.info("Evaluating %d run(s) in experiment '%s'", len(runs), exp_name)

    metrics = {}
    for run in runs:
        try:
            model = create_model(run)
            evaluator = create_evaluator(model, run, experiment_name=exp_name)

            eval_kwargs = {}
            if args.device is not None:
                eval_kwargs["device"] = normalize_device(args.device)
            log.info("Evaluating %s ...", run.run_id)
            result = evaluator.evaluate(**eval_kwargs)
            metrics[run.run_id] = result
            log.info("Evaluation complete: %s - mAP50=%.2f, mAP50-95=%.2f",
                     run.run_id, result.get("map50"), result.get("map50_95"))

        except Exception as e:
            log.error("Evaluating %s failed: %s", run.run_id, e, exc_info=True)
            metrics[run.run_id] = {"error": str(e)}

    return metrics


def _resolve_runs(cfg, args):
    if args.run:
        run = cfg.get_run(args.run)
        if run is None:
            print(f"Error: run_id '{args.run}' not found in config.")
            sys.exit(1)
        return [run]
    return cfg.enabled_runs


def list_enabled_runs(cfg):
    runs = cfg.enabled_runs
    print(f"\n  Enabled runs ({len(runs)}):")
    print(f"  {'─'*70}")
    for r in runs:
        tt = r.get("tiny_train")
        if tt is None:
            tt = cfg.global_config.get("tiny_train", {})
        tt_enabled = tt.get("enabled", False)
        tt_ratio = tt.get("freeze_portion", "—")
        print(f"  {r.run_id:<38} {r.family}{r.variant:<5}  "
              f"epochs={r.training.get('epochs','?'):<3}  "
              f"TinyTrain={'Y' if tt_enabled else 'N'}  "
              f"ratio={tt_ratio}")
    print()


def print_summary(results: dict, phase: str):
    print(f"\n{'='*60}")
    print(f"  {phase} Summary")
    print(f"{'='*60}")
    for run_id, result in results.items():
        is_error = isinstance(result, dict) and "error" in result
        status = "FAILED" if is_error else "OK"
        print(f"  {run_id}: {status}")
    print()


def main():
    args = parse_args()
    cfg = load_config()

    if args.enabled:
        list_enabled_runs(cfg)
        return

    if args.device is None:
        args.device = detect_device()
    log.info("Device: %s", args.device)

    exp_name = resolve_experiment_name(args.name)
    log.info("Experiment: %s", exp_name)

    if args.train:
        train_results = run_training(cfg, args, exp_name)
        print_summary(train_results, "Training")

    if args.evaluate:
        eval_results = run_evaluation(cfg, args, exp_name)
        csv_path = Path(f"outputs/detection/{exp_name}/comparison.csv")
        generate_all_plots(csv_path, Path(f"outputs/detection/{exp_name}"))
        print_summary(eval_results, "Evaluation")


if __name__ == "__main__":
    main()
