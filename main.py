"""
Entry point for the YOLO benchmarking pipeline.

Supports training and evaluation modes. In training mode, it loads
the config and trains all enabled runs. In evaluation mode, it loads
trained weights and benchmarks each model, saving results to
outputs/detection_experiments/.

Usage:
    python main.py --train                          # train all enabled runs
    python main.py --train --run yolov8_nano        # train a single run
    python main.py --train --epochs 10              # override epochs

    python main.py --evaluate                       # evaluate all trained runs
    python main.py --evaluate --run yolov8_nano     # evaluate a single run

    python main.py --train --evaluate               # train then evaluate
"""

import os
import argparse
import sys

# MPS compatibility: prevent "zeros: Dimension size must be non-negative" crashes
# on Apple Silicon. HIGH_WATERMARK_RATIO avoids memory fragmentation, and
# ENABLE_MPS_FALLBACK allows unsupported ops to fall back to CPU gracefully.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from configs.config_loader import load_config
from models import create_model
from training import create_trainer
from evaluation import create_evaluator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO benchmarking pipeline")

    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Run training for enabled models",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Run evaluation on trained models",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Target a specific run_id (e.g. 'yolov8_nano')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs for all runs",
    )

    args = parser.parse_args()
    if not args.train and not args.evaluate:
        parser.print_help()
        print("\n  Error: specify at least one of --train or --evaluate.")
        sys.exit(1)

    return args


def run_training(cfg, args) -> dict:
    """Train all enabled (or targeted) runs and return results."""
    runs = _resolve_runs(cfg, args)
    print(f"\n  Loaded config with {len(runs)} run(s) to train.\n")

    results = {}
    for run in runs:
        try:
            model = create_model(run)
            trainer = create_trainer(model, run)

            override_kwargs = {}
            if args.epochs is not None:
                override_kwargs["epochs"] = args.epochs

            result = trainer.train(**override_kwargs)
            results[run.run_id] = result

        except Exception as e:
            print(f"\n  [ERROR] Training {run.run_id} failed: {e}")
            results[run.run_id] = {"error": str(e)}

    return results


def run_evaluation(cfg, args) -> dict:
    """Evaluate all trained (or targeted) runs and return metrics."""
    runs = _resolve_runs(cfg, args)
    print(f"\n  Loaded config with {len(runs)} run(s) to evaluate.\n")

    metrics = {}
    for run in runs:
        try:
            model = create_model(run)
            evaluator = create_evaluator(model, run)

            result = evaluator.evaluate()
            metrics[run.run_id] = result

        except Exception as e:
            print(f"\n  [ERROR] Evaluating {run.run_id} failed: {e}")
            metrics[run.run_id] = {"error": str(e)}

    return metrics


def _resolve_runs(cfg, args):
    """Get the list of runs to process based on CLI args."""
    if args.run:
        run = cfg.get_run(args.run)
        if run is None:
            print(f"Error: run_id '{args.run}' not found in config.")
            sys.exit(1)
        return [run]
    return cfg.enabled_runs


def print_summary(results: dict, phase: str):
    """Print a clean summary of training/evaluation results."""
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

    if args.train:
        train_results = run_training(cfg, args)
        print_summary(train_results, "Training")

    if args.evaluate:
        eval_results = run_evaluation(cfg, args)
        print_summary(eval_results, "Evaluation")


if __name__ == "__main__":
    main()
