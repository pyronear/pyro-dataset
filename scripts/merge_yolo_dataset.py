"""
Merge wildfire_yolo and fp_yolo into final YOLO datasets.

Combines images and labels from both sources into two output directories:
  - yolo_train_val: train and val splits
  - yolo_test:      test split only

A data.yaml is written in each output directory.

Usage:
    python merge_yolo_dataset.py
    python merge_yolo_dataset.py --dry-run

Arguments:
    --wf-dataset    Path to wildfire_yolo (default: data/processed/wildfire_yolo).
    --fp-dataset    Path to fp_yolo (default: data/processed/fp_yolo).
    --output-train-val  Output directory for train/val (default: data/processed/yolo_train_val).
    --output-test       Output directory for test (default: data/processed/yolo_test).
    --dry-run       Print counts without writing.
    -log, --loglevel  Logging level (default: info).
"""

import argparse
import logging
import shutil
from pathlib import Path


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge WF and FP YOLO datasets.")
    parser.add_argument("--wf-dataset", type=Path, default=Path("data/processed/wildfire_yolo"))
    parser.add_argument("--fp-dataset", type=Path, default=Path("data/processed/fp_yolo"))
    parser.add_argument("--output-train-val", type=Path, default=Path("data/processed/yolo_train_val"))
    parser.add_argument("--output-test", type=Path, default=Path("data/processed/yolo_test"))
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("-log", "--loglevel", default="info")
    return parser


DATA_YAML_TRAIN_VAL = """\
path: .
train: images/train
val: images/val

nc: 1
names:
  - smoke
"""

DATA_YAML_TEST = """\
path: .
test: images/test

nc: 1
names:
  - smoke
"""


def copy_split(sources: list[Path], dst: Path, dry_run: bool) -> int:
    count = 0
    for src in sources:
        if not src.is_dir():
            continue
        for f in src.iterdir():
            if f.name.startswith("."):
                continue
            if not dry_run:
                shutil.copy2(f, dst / f.name)
            count += 1
    return count


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    wf: Path = args["wf_dataset"]
    fp: Path = args["fp_dataset"]
    out_tv: Path = args["output_train_val"]
    out_test: Path = args["output_test"]
    dry_run: bool = args["dry_run"]

    splits_tv = ["train", "val"]
    splits_test = ["test"]

    if not dry_run:
        for split in splits_tv:
            (out_tv / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_tv / "labels" / split).mkdir(parents=True, exist_ok=True)
        for split in splits_test:
            (out_test / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_test / "labels" / split).mkdir(parents=True, exist_ok=True)

    counters = {}
    for split in splits_tv + splits_test:
        out = out_tv if split in splits_tv else out_test
        for sub in ["images", "labels"]:
            sources = [wf / sub / split, fp / sub / split]
            dst = out / sub / split
            n = copy_split(sources, dst, dry_run)
            counters[f"{split}/{sub}"] = n

    if not dry_run:
        (out_tv / "data.yaml").write_text(DATA_YAML_TRAIN_VAL)
        (out_test / "data.yaml").write_text(DATA_YAML_TEST)

    print(f"\n{'='*50}")
    print(f"{'DRY RUN — ' if dry_run else ''}Final YOLO dataset")
    print(f"\n  yolo_train_val/")
    for split in splits_tv:
        n_img = counters[f"{split}/images"]
        n_lbl = counters[f"{split}/labels"]
        print(f"    {split:<6}: {n_img} images, {n_lbl} labels")
    print(f"\n  yolo_test/")
    for split in splits_test:
        n_img = counters[f"{split}/images"]
        n_lbl = counters[f"{split}/labels"]
        print(f"    {split:<6}: {n_img} images, {n_lbl} labels")
    print(f"{'='*50}\n")

    if dry_run:
        print("Dry run — nothing written.")
