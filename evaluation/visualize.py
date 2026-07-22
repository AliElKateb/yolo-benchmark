"""
Generate comparison plots from experiment results CSV.

Usage (called automatically from main.py after evaluation):
    from evaluation.visualize import generate_all_plots
    generate_all_plots("outputs/detection/exp_001/comparison.csv", "outputs/detection/exp_001")
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#4A72B5", "#E8856C", "#5CB85C", "#9467BD", "#F0AD4E", "#D9534F"]
GRAY = "#6C757D"
LIGHT_GRAY = "#E0E0E0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.facecolor": "white",
    "axes.edgecolor": LIGHT_GRAY,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.color": LIGHT_GRAY,
    "grid.linestyle": "-",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _model_label(run_id: str) -> str:
    return run_id.replace("yolov5_", "v5 ").replace("yolov8_", "v8 ")


def _plot_bar(ax, labels, groups, ylabel, palette=None):
    if palette is None:
        palette = PALETTE
    x = np.arange(len(labels))
    n = len(groups)
    w = 0.65 / n
    for i, (name, vals) in enumerate(groups):
        offset = (i - (n - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=palette[i % len(palette)], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, facecolor="white", edgecolor=LIGHT_GRAY, fontsize=9)
    ax.set_axisbelow(True)


def _plot_map_comparison(data, output_dir):
    labels = [_model_label(r["Run ID"]) for r in data]
    map50 = [float(r["mAP50"]) for r in data]
    map95 = [float(r["mAP50-95"]) for r in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_bar(ax, labels, [("mAP50", map50), ("mAP50-95", map95)], "mAP (%)")
    ax.set_title("Mean Average Precision")
    fig.savefig(output_dir / "mAP_comparison.png")
    plt.close(fig)


def _plot_precision_recall_f1(data, output_dir):
    labels = [_model_label(r["Run ID"]) for r in data]
    p = [float(r["Precision"]) for r in data]
    r = [float(r["Recall"]) for r in data]
    f1 = [float(r["F1-Score"]) for r in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_bar(ax, labels, [("Precision", p), ("Recall", r), ("F1", f1)], "Score (%)")
    ax.set_title("Precision, Recall & F1 Score")
    fig.savefig(output_dir / "precision_recall_f1.png")
    plt.close(fig)


def _plot_inference_speed(data, output_dir):
    labels = [_model_label(r["Run ID"]) for r in data]
    inf = [float(r["Inference (ms)"]) for r in data]
    total = [float(r["Total (ms/img)"]) for r in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    w = 0.3
    ax.bar(x - w / 2, inf, w, label="Inference", color=PALETTE[0], zorder=3)
    ax.bar(x + w / 2, total, w, label="Total", color=PALETTE[1], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Inference Speed  (lower is better)")
    ax.legend(frameon=True, facecolor="white", edgecolor=LIGHT_GRAY, fontsize=9)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "inference_speed.png")
    plt.close(fig)


def _plot_model_size(data, output_dir):
    labels = [_model_label(r["Run ID"]) for r in data]
    params = [float(r["Parameters"]) for r in data]
    size_mb = [float(r["Size (MB)"]) for r in data]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    x = np.arange(len(labels))
    w = 0.3
    ax1.bar(x - w / 2, [p / 1e6 for p in params], w, label="Parameters (M)", color=PALETTE[0], zorder=3)
    ax1.set_ylabel("Parameters (M)")

    ax2.plot(x + w / 2, size_mb, "o-", color=PALETTE[1], linewidth=2, markersize=6, label="Size (MB)", zorder=4)
    ax2.set_ylabel("Size (MB)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Model Size & Parameters")

    lines = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[0]),
             plt.Line2D([0], [0], color=PALETTE[1], linewidth=2, marker="o")]
    ax1.legend(lines, ["Parameters (M)", "Size (MB)"],
               frameon=True, facecolor="white", edgecolor=LIGHT_GRAY, fontsize=9, loc="upper left")
    ax1.set_axisbelow(True)
    fig.savefig(output_dir / "model_size.png")
    plt.close(fig)


def _plot_summary_dashboard(data, output_dir):
    labels = [_model_label(r["Run ID"]) for r in data]
    keys = [
        ("mAP50",         "mAP50"),
        ("mAP50-95",      "mAP50-95"),
        ("Precision",     "Precision"),
        ("Recall",        "Recall"),
        ("F1-Score",      "F1"),
        ("Inference (ms)", "Inference\n(ms)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    for ax, (col, short) in zip(axes.flat, keys):
        vals = [float(r[col]) for r in data]
        ax.bar(labels, vals, color=colors, zorder=3, width=0.6)
        ax.set_title(short, fontsize=10, fontweight="semibold")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.suptitle("Experiment Summary Dashboard", fontsize=13, y=1.01)
    fig.savefig(output_dir / "summary_dashboard.png")
    plt.close(fig)


def _plot_radar_chart(data, output_dir):
    labels = [_model_label(r["Run ID"]) for r in data]
    metrics_keys = ["mAP50", "mAP50-95", "Precision", "Recall", "F1-Score"]
    metrics_short = ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]

    raw = np.array([[float(r[k]) for k in metrics_keys] for r in data])
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (raw - mins) / ranges * 100

    n_metrics = len(metrics_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]

    for i, lbl in enumerate(labels):
        vals = normalized[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", label=lbl, color=colors[i], linewidth=1.5, markersize=4)
        ax.fill(angles, vals, alpha=0.06, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_short, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color=GRAY)
    ax.set_title("Normalized Metrics Comparison\n(higher = better)", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True,
              facecolor="white", edgecolor=LIGHT_GRAY, fontsize=8)
    ax.grid(True, alpha=0.3, color=LIGHT_GRAY)
    fig.savefig(output_dir / "radar_chart.png")
    plt.close(fig)


def generate_all_plots(csv_path: str | Path, output_dir: str | Path):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if not csv_path.exists():
        print(f"  [SKIP] comparison.csv not found at {csv_path}")
        return

    data = _load_csv(csv_path)
    if not data:
        print(f"  [SKIP] comparison.csv is empty")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Generating plots in {plots_dir} ...")

    _plot_map_comparison(data, plots_dir)
    _plot_precision_recall_f1(data, plots_dir)
    _plot_inference_speed(data, plots_dir)
    _plot_model_size(data, plots_dir)
    _plot_summary_dashboard(data, plots_dir)
    _plot_radar_chart(data, plots_dir)

    print(f"  Plots saved: {len(list(plots_dir.glob('*.png')))} files")
