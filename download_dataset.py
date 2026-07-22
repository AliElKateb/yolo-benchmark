"""
Download datasets from Roboflow.

Datasets:
  - SMD Components (dainius/smdcomponents v6) — 7.8k images, 4 classes
  - Mechanical tools-10000 (mechanical-tools/mechanical-tools-10000 v3) — 9.3k images, 5 classes

Usage:
    export RF_API_KEY="your_key"
    python download_dataset.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
DATASET_DIR = "dataset"


def _dataset_exists(path: str) -> bool:
    return Path(path, "data.yaml").exists()


def download_smd_components(rf: Roboflow) -> str:
    dest = f"{DATASET_DIR}/smd_components"
    if _dataset_exists(dest):
        print(f"Dataset already exists at '{dest}/' — skipping download.")
        return dest
    print(f"Downloading SMD Components to '{dest}/' ...")
    project = rf.workspace("dainius").project("smdcomponents")
    version = project.version(6)
    dataset = version.download("yolov8", location=dest)
    return dataset.location


def download_mechanical_tools(rf: Roboflow) -> str:
    dest = f"{DATASET_DIR}/mechanical_tools"
    if _dataset_exists(dest):
        print(f"Dataset already exists at '{dest}/' — skipping download.")
        return dest
    print(f"Downloading Mechanical tools-10000 to '{dest}/' ...")
    project = rf.workspace("mechanical-tools").project("mechanical-tools-10000")
    version = project.version(3)
    dataset = version.download("yolov8", location=dest)
    return dataset.location


def download_drillbit(rf: Roboflow) -> str:
    dest = f"{DATASET_DIR}/drillbit_detection"
    if _dataset_exists(dest):
        print(f"Dataset already exists at '{dest}/' — skipping download.")
        return dest
    print(f"Downloading Drillbit Detection to '{dest}/' ...")
    project = rf.workspace("small-objects-detection").project("drillbit-detection")
    version = project.version(3)
    dataset = version.download("yolov8", location=dest)
    return dataset.location


def main():
    api_key = os.getenv("RF_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RF_API_KEY not set. Create a .env file with:\n"
            "  RF_API_KEY=your_roboflow_api_key"
        )

    rf = Roboflow(api_key=api_key)

    smd_path = download_smd_components(rf)
    mech_path = download_mechanical_tools(rf)
    drillbit_path = download_drillbit(rf)

    print(f"\nDone. Datasets saved to:")
    print(f"  {smd_path}")
    print(f"  {mech_path}")
    print(f"  {drillbit_path}")


if __name__ == "__main__":
    main()
