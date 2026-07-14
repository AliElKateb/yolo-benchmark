"""
Entry point for the YOLO benchmarking pipeline.

Supports training and evaluation modes. In training mode, it loads
the config and trains all enabled runs. In evaluation mode, it loads
trained weights and benchmarks each model, saving results to
outputs/detection/.

Usage:
    python main.py --train --name exp_001                        # train all enabled runs
    python main.py --train --run yolov8_nano --name exp_001      # train a single run
    python main.py --train --epochs 10 --name exp_001            # override epochs
    python main.py --evaluate --name exp_001                     # evaluate all trained runs
    python main.py --evaluate --run yolov8_nano --name exp_001   # evaluate a single run
    python main.py --train --evaluate --name exp_001             # train then evaluate
"""

import os
import re
import argparse
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from configs.config_loader import load_config
from models import create_model
from training import create_trainer
from evaluation import create_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO benchmarking pipeline")

    parser.add_argument("--train", action="store_true", default=False, help="Run training for enabled models")
    parser.add_argument("--evaluate", action="store_true", default=False, help="Run evaluation on trained models")
    parser.add_argument("--run", type=str, default=None, help="Target a specific run_id (e.g. 'yolov8_nano')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs for all runs")
    parser.add_argument("--name", type=str, default=None, help="Experiment folder name (auto-generated if omitted)")

    args = parser.parse_args()
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
    print(f"  Loaded config with {len(runs)} run(s) to train.\n")

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

            result = trainer.train(**override_kwargs)
            results[run.run_id] = result

        except Exception as e:
            print(f"\n  [ERROR] Training {run.run_id} failed: {e}")
            results[run.run_id] = {"error": str(e)}

    return results


def run_evaluation(cfg, args, exp_name: str) -> dict:
    runs = _resolve_runs(cfg, args)
    print(f"  Loaded config with {len(runs)} run(s) to evaluate.\n")

    metrics = {}
    for run in runs:
        try:
            model = create_model(run)
            evaluator = create_evaluator(model, run, experiment_name=exp_name)

            result = evaluator.evaluate()
            metrics[run.run_id] = result

        except Exception as e:
            print(f"\n  [ERROR] Evaluating {run.run_id} failed: {e}")
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

    exp_name = resolve_experiment_name(args.name)
    print(f"\n  Experiment: {exp_name}")

    if args.train:
        train_results = run_training(cfg, args, exp_name)
        print_summary(train_results, "Training")

    if args.evaluate:
        eval_results = run_evaluation(cfg, args, exp_name)
        print_summary(eval_results, "Evaluation")


if __name__ == "__main__":
    main()
