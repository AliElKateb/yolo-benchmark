"""
Entry point for the YOLO benchmarking pipeline.

Supports the two-stage baseline/retrain workflow:
  1. --train  trains the baseline split (70%) and evaluates on shared valid/.
  2. --retrain continues from a baseline checkpoint on the retrain split (30%),
     optionally freezing filters with TinyTrain.
  3. --merge   aggregates baseline + retrain comparison CSVs into one table,
     ranks the runs by a composite score and saves ranking.csv.

Usage:
    python main.py --train --name exp_001                          # train all enabled runs
    python main.py --train --run mech_yolov5_small --name exp_001  # train a single run
    python main.py --train --epochs 10 --device mps --name exp_001 # override epochs + device
    python main.py --evaluate --name exp_001                       # evaluate all trained runs
    python main.py --retrain --name exp_001 --strategy tiny_train --freeze-ratio 0.5
    python main.py --retrain --name exp_001 --strategy whole
    python main.py --merge --name exp_001 --merge-name exp_combined
    python main.py --enabled                                        # list enabled runs and exit
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import yaml

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

RETRAIN_STRATEGIES = ("tiny_train", "whole")


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
    parser = argparse.ArgumentParser(
        description="YOLO benchmarking pipeline: baseline training -> retraining -> merge comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Train baseline (70% split) and auto-evaluate on shared valid/:\n"
            "    python main.py --train --name exp_001\n\n"
            "  Retrain from the baseline with TinyTrain (freeze 90%) on the 30% split:\n"
            "    python main.py --retrain --name exp_001 --strategy tiny_train --freeze-ratio 0.9\n\n"
            "  Retrain the whole model (no freezing):\n"
            "    python main.py --retrain --name exp_001 --strategy whole\n\n"
            "  Merge the baseline + all its retrain experiments into one table:\n"
            "    python main.py --merge --name exp_001\n\n"
            "  --merge scans outputs/detection/exp_001/ (baseline) and every\n"
            "  outputs/detection/exp_001_retrain_*/ (its retrains) comparison.csv,\n"
            "  tags the run ids (base_, tt0.90_, tt0.50_, whole_, ...), ranks the\n"
            "  runs by a composite score (accuracy + speed + size) and writes the\n"
            "  combined CSV, a ranking.csv and plots to outputs/detection/exp_001_merged/.\n"
        ),
    )

    parser.add_argument("--train", action="store_true", default=False,
                        help="Train the baseline split (70%%) for enabled models, then auto-evaluate on shared valid/")
    parser.add_argument("--retrain", action="store_true", default=False,
                        help="Retrain a baseline experiment on the retrain split (30%%)")
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help="Evaluate trained models on the shared valid split")
    parser.add_argument("--merge", action="store_true", default=False,
                        help="Merge the baseline experiment's comparison.csv with all its retrain experiment(s) into one table + ranking + plots (requires --name; does not train/evaluate)")
    parser.add_argument("--strategy", type=str, choices=RETRAIN_STRATEGIES, default=None,
                        help="Retrain strategy: 'tiny_train' (freeze least-important filters) or 'whole' (retrain all)")
    parser.add_argument("--run", type=str, default=None, help="Target a specific run_id (e.g. 'mech_yolov5_small')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs for all runs")
    parser.add_argument("--device", type=str, default=None, help="Override device for all runs (e.g. 'cpu', 'mps', 'cuda:0')")
    parser.add_argument("--freeze-ratio", type=float, default=None,
                        help="TinyTrain freeze_portion (0.0-1.0); required for --strategy tiny_train")
    parser.add_argument("--enabled", action="store_true", default=False, help="List enabled runs and exit")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment folder name (auto-generated for --train/--evaluate; REQUIRED for --retrain/--merge)")
    parser.add_argument("--merge-name", type=str, default=None,
                        help="Output folder for --merge (default: <name>_merged)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Replace the default 'base_' prefix on baseline run ids in the --merge output")

    args = parser.parse_args()

    if args.enabled:
        return args

    if args.merge:
        if not args.name:
            parser.print_help()
            print("\n  Error: --merge requires --name (the baseline experiment).")
            sys.exit(1)
        return args

    if args.train and args.retrain:
        print("\n  Error: --train and --retrain cannot be combined. Use one stage at a time.")
        sys.exit(1)

    if not args.train and not args.evaluate and not args.retrain:
        parser.print_help()
        print("\n  Error: specify at least one of --train, --retrain or --evaluate.")
        sys.exit(1)

    if args.retrain:
        if not args.name:
            print("\n  Error: --retrain requires --name (the baseline experiment to continue from).")
            sys.exit(1)
        if not args.strategy:
            print("\n  Error: --retrain requires --strategy {'" + "', '".join(RETRAIN_STRATEGIES) + "'}.")
            sys.exit(1)
        if args.strategy == "tiny_train":
            if args.freeze_ratio is None:
                print("\n  Error: --strategy tiny_train requires --freeze-ratio (0.0-1.0).")
                sys.exit(1)
            if not (0.0 <= args.freeze_ratio <= 1.0):
                print(f"\n  Error: --freeze-ratio must be between 0.0 and 1.0, got {args.freeze_ratio}.")
                sys.exit(1)
    elif args.freeze_ratio is not None and not args.train:
        print("\n  Error: --freeze-ratio only applies to retraining (--retrain --strategy tiny_train).")
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


def _resolve_runs(cfg, args):
    if args.run:
        run = cfg.get_run(args.run)
        if run is None:
            print(f"Error: run_id '{args.run}' not found in config.")
            sys.exit(1)
        return [run]
    return cfg.enabled_runs


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
                "data": run.baseline_data,
                "tiny_train_enabled": False,
            }
            if args.epochs is not None:
                override_kwargs["epochs"] = args.epochs
            if args.device is not None:
                override_kwargs["device"] = normalize_device(args.device)

            log.info("Training baseline %s (family=%s, variant=%s, data=%s)",
                     run.run_id, run.family, run.variant, run.baseline_data)
            result = trainer.train(**override_kwargs)
            results[run.run_id] = result
            log.info("Training complete: %s", run.run_id)

        except Exception as e:
            log.error("Training %s failed: %s", run.run_id, e, exc_info=True)
            results[run.run_id] = {"error": str(e)}

    return results


def _find_baseline_weights(exp_name: str, run) -> Path | None:
    candidates = [
        Path(f"outputs/detection/{exp_name}/{run.run_id}/weights/best.pt"),
        Path(f"outputs/detection/{exp_name}/{run.run_id}/weights/last.pt"),
        Path(f"outputs/detection/{exp_name}/{run.training.get('name', run.run_id)}/weights/best.pt"),
        Path(f"outputs/detection/{exp_name}/{run.training.get('name', run.run_id)}/weights/last.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _retrain_suffix(strategy: str, freeze_ratio: float | None) -> str:
    if strategy == "tiny_train":
        return f"tt_{freeze_ratio:.2f}"
    return "whole"


def _write_retrain_args(retrain_exp: str, base_exp: str, strategy: str, freeze_ratio: float | None,
                        runs: list, run_freeze: dict) -> Path:
    payload = {
        "strategy": strategy,
        "freeze_ratio": freeze_ratio,
        "baseline_experiment": base_exp,
        "retrain_experiment": retrain_exp,
        "epochs": runs[0].training.get("epochs") if runs else None,
        "runs": {r.run_id: run_freeze.get(r.run_id, {}) for r in runs},
    }
    out = Path(f"outputs/detection/{retrain_exp}/argument.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"  Retrain arguments saved to {out}")
    return out


def run_retraining(cfg, args, base_exp: str) -> str:
    runs = _resolve_runs(cfg, args)
    strategy = args.strategy
    retrain_exp = f"{base_exp}_retrain_{_retrain_suffix(strategy, args.freeze_ratio)}"
    log.info("Retraining %d run(s) from '%s' -> '%s' (strategy=%s)",
             len(runs), base_exp, retrain_exp, strategy)

    results = {}
    run_freeze = {}
    for run in runs:
        try:
            baseline_pt = _find_baseline_weights(base_exp, run)
            if baseline_pt is None:
                print(f"\n  Error: no baseline weights found for '{run.run_id}' in experiment '{base_exp}'.")
                results[run.run_id] = {"error": "baseline weights not found"}
                continue

            model = create_model(run)
            trainer = create_trainer(model, run)

            override_kwargs = {
                "project": f"outputs/detection/{retrain_exp}",
                "name": run.run_id,
                "exist_ok": True,
                "data": run.retrain_data,
                "weights": str(baseline_pt),
                "tiny_train_enabled": (strategy == "tiny_train"),
            }
            if strategy == "tiny_train":
                override_kwargs["freeze_ratio"] = args.freeze_ratio
            if args.epochs is not None:
                override_kwargs["epochs"] = args.epochs
            if args.device is not None:
                override_kwargs["device"] = normalize_device(args.device)

            log.info("Retraining %s from %s (data=%s)",
                     run.run_id, baseline_pt, run.retrain_data)
            result = trainer.train(**override_kwargs)
            results[run.run_id] = result

            freeze = model.last_freeze_list
            info = {"baseline_weights": str(baseline_pt)}
            if strategy == "tiny_train" and freeze:
                info["freeze_layers"] = len(freeze)
                info["freeze_filters"] = sum(len(entry) - 1 for entry in freeze)
                info["freeze_list"] = freeze
            run_freeze[run.run_id] = info
            log.info("Retraining complete: %s", run.run_id)

        except Exception as e:
            log.error("Retraining %s failed: %s", run.run_id, e, exc_info=True)
            results[run.run_id] = {"error": str(e)}

    _write_retrain_args(retrain_exp, base_exp, strategy, args.freeze_ratio, runs, run_freeze)
    print_summary(results, f"Retraining ({strategy})")

    print(f"\n  Auto-evaluating retrained models on shared valid/ ...")
    eval_results = run_evaluation(cfg, args, retrain_exp)
    csv_path = Path(f"outputs/detection/{retrain_exp}/comparison.csv")
    generate_all_plots(csv_path, Path(f"outputs/detection/{retrain_exp}"))
    print_summary(eval_results, "Evaluation")

    return retrain_exp


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


def _retrain_dir_prefix(exp_name: str, base: str) -> str | None:
    if exp_name == base:
        return None
    marker = f"{base}_retrain_"
    if not exp_name.startswith(marker):
        return None
    suffix = exp_name[len(marker):]
    if suffix.startswith("tt_"):
        return f"tt{suffix[3:]}"
    if suffix == "whole":
        return "whole"
    return suffix


RANKING_WEIGHTS = {
    "mAP50-95": 0.30,
    "mAP50": 0.15,
    "Precision": 0.10,
    "Recall": 0.10,
    "F1-Score": 0.10,
    "Inference (ms)": 0.15,
    "Size (MB)": 0.10,
}


def _composite_score(rows: list[dict]) -> None:
    cols = list(RANKING_WEIGHTS)
    for row in rows:
        for c in cols:
            row[f"_n_{c}"] = float(row.get(c, 0) or 0)
    lo = {c: min(r[f"_n_{c}"] for r in rows) for c in cols}
    hi = {c: max(r[f"_n_{c}"] for r in rows) for c in cols}
    for r in rows:
        score = 0.0
        for c, w in RANKING_WEIGHTS.items():
            rng = hi[c] - lo[c]
            if rng == 0:
                norm = 1.0
            elif c in ("Inference (ms)", "Size (MB)"):
                norm = (hi[c] - r[f"_n_{c}"]) / rng
            else:
                norm = (r[f"_n_{c}"] - lo[c]) / rng
            score += norm * w
        r["_score"] = score * 100
    for r in rows:
        for c in cols:
            r.pop(f"_n_{c}", None)


def _freeze_ratio_value(run_id: str) -> float:
    """Extract the TinyTrain freeze ratio from a tagged run id, or NaN if unknown."""
    tag = run_id.split("_", 1)[0].lower()
    if tag.startswith("tt"):
        try:
            return float(tag[2:])
        except ValueError:
            return float("nan")
    if tag == "whole":
        return 0.0
    return float("nan")


def _freeze_ratio_label(run_id: str) -> str:
    val = _freeze_ratio_value(run_id)
    if val != val:  # NaN
        return "—"
    return f"{val * 100:g}%"


def _rank_and_report(rows: list[dict], merged_dir: Path, base_prefix: str) -> None:
    _composite_score(rows)
    ranked = sorted(rows, key=lambda r: r["_score"], reverse=True)
    baseline = next((r for r in rows if r["Run ID"].startswith(f"{base_prefix}_")), None)
    base_score = baseline["_score"] if baseline else None

    print(f"\n  Overall ranking (composite of mAP50-95, mAP50, P/R/F1, speed, size; higher = better):")
    header = (f"  {'Rank':<5}{'Run ID':<40}{'Freeze':>8}{'mAP50-95':>9}{'mAP50':>7}"
              f"{'F1':>7}{'Inf(ms)':>8}{'Score':>7}{'Δ vs base':>11}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, r in enumerate(ranked, 1):
        delta = f"{r['_score'] - base_score:+5.1f}" if base_score is not None else ""
        print(f"  {i:<5}{r['Run ID']:<40}{_freeze_ratio_label(r['Run ID']):>8}"
              f"{float(r['mAP50-95']):>9.2f}{float(r['mAP50']):>7.2f}"
              f"{float(r['F1-Score']):>7.2f}{float(r['Inference (ms)']):>8.2f}{r['_score']:>7.1f}{delta:>11}")

    best = ranked[0]
    best_acc = max(rows, key=lambda r: float(r["mAP50-95"]))
    fastest = min(rows, key=lambda r: float(r["Inference (ms)"]))
    print(f"  Best overall : {best['Run ID']} (score {best['_score']:.1f})")
    print(f"  Best accuracy: {best_acc['Run ID']} (mAP50-95 {float(best_acc['mAP50-95']):.2f})")
    print(f"  Fastest      : {fastest['Run ID']} ({float(fastest['Inference (ms)']):.2f} ms)")

    rank_csv = merged_dir / "ranking.csv"
    with open(rank_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Rank", "Run ID", "Experiment", "Freeze Ratio", "Score", "Delta vs Baseline",
            "mAP50", "mAP50-95", "Precision", "Recall", "F1-Score",
            "Inference (ms)", "Total (ms/img)", "Parameters", "Size (MB)",
        ])
        writer.writeheader()
        for i, r in enumerate(ranked, 1):
            writer.writerow({
                "Rank": i,
                "Run ID": r["Run ID"],
                "Experiment": r.get("Experiment", ""),
                "Freeze Ratio": _freeze_ratio_value(r["Run ID"]),
                "Score": f"{r['_score']:.1f}",
                "Delta vs Baseline": f"{r['_score'] - base_score:+.1f}" if base_score is not None else "",
                "mAP50": r.get("mAP50", ""),
                "mAP50-95": r.get("mAP50-95", ""),
                "Precision": r.get("Precision", ""),
                "Recall": r.get("Recall", ""),
                "F1-Score": r.get("F1-Score", ""),
                "Inference (ms)": r.get("Inference (ms)", ""),
                "Total (ms/img)": r.get("Total (ms/img)", ""),
                "Parameters": r.get("Parameters", ""),
                "Size (MB)": r.get("Size (MB)", ""),
            })
    print(f"  Ranking saved to {rank_csv}")


def run_merge(cfg, args) -> None:
    base = args.name
    base_csv = Path(f"outputs/detection/{base}/comparison.csv")
    if not base_csv.exists():
        print(f"\n  Error: baseline experiment '{base}' has no comparison.csv (run --train first).")
        sys.exit(1)

    sources = [(base, args.tag or "base")]
    for d in sorted(Path("outputs/detection").iterdir()):
        if not d.is_dir():
            continue
        prefix = _retrain_dir_prefix(d.name, base)
        if prefix is not None and (d / "comparison.csv").exists():
            sources.append((d.name, prefix))

    merged_name = args.merge_name or f"{base}_merged"
    merged_dir = Path(f"outputs/detection/{merged_name}")
    merged_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    for exp, prefix in sources:
        csv_path = Path(f"outputs/detection/{exp}/comparison.csv")
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if not row or not row.get("Run ID"):
                    continue
                tagged = f"{prefix}_{row['Run ID']}"
                if tagged in seen:
                    continue
                seen.add(tagged)
                new_row = dict(row)
                new_row["Run ID"] = tagged
                new_row["Experiment"] = exp
                rows.append(new_row)

    if not rows:
        print(f"\n  Error: no rows found to merge.")
        sys.exit(1)

    fieldnames = ["Run ID", "Experiment"] + [k for k in rows[0] if k not in ("Run ID", "Experiment")]
    merged_csv = merged_dir / "comparison.csv"
    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\n  Merged {len(rows)} rows from {len(sources)} experiment(s) into {merged_csv}")
    print(f"    Sources:")
    for exp, prefix in sources:
        print(f"      {exp:<45} prefix='{prefix}'")
    _rank_and_report(rows, merged_dir, args.tag or "base")
    generate_all_plots(merged_csv, merged_dir)


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
              f"base={Path(r.baseline_data).parent.name:<18}  "
              f"retrain={Path(r.retrain_data).parent.name:<18}  "
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

    if args.merge:
        run_merge(cfg, args)
        return

    if args.device is None:
        args.device = detect_device()
    log.info("Device: %s", args.device)

    if args.retrain:
        run_retraining(cfg, args, args.name)
        return

    exp_name = resolve_experiment_name(args.name)
    log.info("Experiment: %s", exp_name)

    if args.train:
        train_results = run_training(cfg, args, exp_name)
        print_summary(train_results, "Training")
        print("\n  Auto-evaluating trained models on shared valid/ ...")
        eval_results = run_evaluation(cfg, args, exp_name)
        csv_path = Path(f"outputs/detection/{exp_name}/comparison.csv")
        generate_all_plots(csv_path, Path(f"outputs/detection/{exp_name}"))
        print_summary(eval_results, "Evaluation")

    if args.evaluate:
        eval_results = run_evaluation(cfg, args, exp_name)
        csv_path = Path(f"outputs/detection/{exp_name}/comparison.csv")
        generate_all_plots(csv_path, Path(f"outputs/detection/{exp_name}"))
        print_summary(eval_results, "Evaluation")


if __name__ == "__main__":
    main()
