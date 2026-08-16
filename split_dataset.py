"""
Split ONLY the training set of a YOLO dataset into two disjoint splits:

  - <name>_baseline/  (default 70% of the original train set)
  - <name>_retrain/   (default 30% of the original train set)

The original test/ and valid/ sets are copied unchanged into BOTH new
datasets, so baseline and retrain share the exact same test and validation
sets (only the training images differ).

Usage:
    python split_dataset.py --name mechanical_tools
    python split_dataset.py --name mechanical_tools --baseline 0.7 --seed 42
    python split_dataset.py --name mechanical_tools --dry-run
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REPO_ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a YOLO dataset's training set into baseline/retrain splits")
    parser.add_argument("--name", type=str, required=True,
                        help="Dataset folder name under dataset/ (e.g. 'mechanical_tools')")
    parser.add_argument("--baseline", type=float, default=0.7,
                        help="Fraction of the training set used for the baseline split (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--force", action="store_true",
                        help="Recreate output directories even if they already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute counts and print the plan without writing files")
    args = parser.parse_args()

    if not (0.0 < args.baseline < 1.0):
        print(f"Error: --baseline must be between 0 and 1, got {args.baseline}")
        sys.exit(1)
    return args


def _iter_pairs(split_dir: Path):
    """Yield (image_path, label_path) pairs for a YOLO images/labels directory."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        return
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        yield img, label if label.exists() else None


def _read_classes(label_path: Path | None) -> frozenset[int]:
    if label_path is None or not label_path.exists():
        return frozenset()
    classes = set()
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            classes.add(int(line.split()[0]))
    return frozenset(classes)


def _write_data_yaml(out_dir: Path, data_cfg: dict):
    has_test = (out_dir / "test" / "images").is_dir()
    has_val = (out_dir / "valid" / "images").is_dir()
    content = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
    }
    if has_val:
        content["val"] = "valid/images"
    if has_test:
        content["test"] = "test/images"
    content["nc"] = data_cfg.get("nc")
    content["names"] = data_cfg.get("names")
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(content, f, sort_keys=False)


def _copy_pair(img: Path, label: Path | None, dst_images: Path, dst_labels: Path):
    dst_images.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dst_images / img.name)
    if label is not None and label.exists():
        dst_labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label, dst_labels / label.name)


def _copy_split(src: Path, out_root: Path, subset: str, label: str) -> int:
    """Copy a whole train/test/valid split (images + labels) into out_root."""
    if not (src / "images").is_dir():
        return 0
    n = 0
    for img, lab in _iter_pairs(src):
        _copy_pair(img, lab, out_root / subset / "images", out_root / subset / "labels")
        n += 1
    print(f"    {out_root.name}/{subset}/  {n} images (shared from dataset/{label})")
    return n


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    dataset_dir = REPO_ROOT / "dataset" / args.name
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at {data_yaml}")
        sys.exit(1)

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    path_base = Path(data_cfg.get("path", str(dataset_dir))).resolve()

    def resolve_split(key):
        rel = data_cfg.get(key)
        if not rel:
            return None
        p = Path(rel)
        if not p.is_absolute():
            p = path_base / p
        return p.resolve()

    train_dir = resolve_split("train")
    test_dir = resolve_split("test")
    valid_dir = resolve_split("val") or resolve_split("valid")

    if train_dir is None or not train_dir.is_dir():
        print(f"Error: train images dir not found: {train_dir}")
        sys.exit(1)

    train_pairs = list(_iter_pairs(train_dir.parent))
    if not train_pairs:
        print(f"Error: no training images found under {train_dir.parent}")
        sys.exit(1)

    groups = {}
    for img, label in train_pairs:
        signature = _read_classes(label)
        groups.setdefault(signature, []).append((img, label))

    baseline_train = []
    retrain_train = []
    for signature, group in groups.items():
        rng.shuffle(group)
        n_baseline = int(round(len(group) * args.baseline))
        baseline_train.extend(group[:n_baseline])
        retrain_train.extend(group[n_baseline:])

    out_baseline = REPO_ROOT / "dataset" / f"{args.name}_baseline"
    out_retrain = REPO_ROOT / "dataset" / f"{args.name}_retrain"

    print(f"\n  Dataset:        {dataset_dir.name}")
    print(f"  Original train: {len(train_pairs)} images (only this set is split)")
    print(f"  Baseline train: {len(baseline_train)} images")
    print(f"  Retrain train:  {len(retrain_train)} images")
    print(f"  Shared test:    {'yes' if test_dir else 'none found'}")
    print(f"  Shared valid:   {'yes' if valid_dir else 'none found'}")

    if args.dry_run:
        print("\n  [DRY RUN] Would write:")
        for out in (out_baseline, out_retrain):
            print(f"    {out}/  (train + shared test/ + shared valid/)")
        print("  No files written.")
        return

    for out in (out_baseline, out_retrain):
        if out.exists() and not args.force:
            print(f"\n  Error: {out} already exists. Use --force to recreate it.")
            sys.exit(1)

    for out, items in ((out_baseline, baseline_train), (out_retrain, retrain_train)):
        if out.exists():
            shutil.rmtree(out)
        print(f"\n  Building {out.name}/")
        for img, label in items:
            _copy_pair(img, label, out / "train" / "images", out / "train" / "labels")
        print(f"    {out.name}/train/  {len(items)} images (split from dataset/{args.name})")
        if test_dir is not None:
            _copy_split(test_dir.parent, out, "test", f"{args.name}/test")
        if valid_dir is not None:
            _copy_split(valid_dir.parent, out, "valid", f"{args.name}/valid")
        _write_data_yaml(out, data_cfg)

    print(f"\n  Done. Both datasets share the same test/ and valid/ sets.")
    print(f"    Baseline: {out_baseline}/data.yaml")
    print(f"    Retrain:  {out_retrain}/data.yaml")


if __name__ == "__main__":
    main()
