"""
Build toy (subsampled) versions of yolo_train_val and sequential_train_val.

Useful for quickly testing training pipelines without the full dataset.
Samples a fixed ratio of images/sequences per split, preserving train/val proportions.

Output structure:
    <yolo-output>/          (mirrors yolo_train_val)
      images/train/
      images/val/
      labels/train/
      labels/val/
      data.yaml

    <sequential-output>/    (mirrors sequential_train_val)
      train/wildfire/<sequence>/
        images/
        labels/
      train/fp/<sequence>/
        ...
      val/wildfire/<sequence>/
        ...
      val/fp/<sequence>/
        ...

Usage:
    python build_toy_dataset.py
    python build_toy_dataset.py --ratio 0.05 --dry-run

Arguments:
    --yolo-src          Source yolo_train_val directory (default: data/processed/yolo_train_val).
    --seq-src           Source sequential_train_val directory (default: data/processed/sequential_train_val).
    --yolo-output       Output yolo toy directory (default: data/processed/yolo_toy).
    --seq-output        Output sequential toy directory (default: data/processed/sequential_toy).
    --ratio             Fraction of data to sample (default: 0.05).
    --random-seed       Random seed (default: 0).
    --dry-run           Print counts without writing.
    -log, --loglevel    Logging level (default: info).
"""

import argparse
import logging
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "val"]


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build toy subsampled datasets.")
    parser.add_argument("--yolo-src", type=Path, default=Path("data/processed/yolo_train_val"))
    parser.add_argument("--seq-src", type=Path, default=Path("data/processed/sequential_train_val"))
    parser.add_argument("--yolo-output", type=Path, default=Path("data/processed/yolo_toy"))
    parser.add_argument("--seq-output", type=Path, default=Path("data/processed/sequential_toy"))
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("-log", "--loglevel", default="info")
    return parser


def sample(items: list, ratio: float, rng: random.Random) -> list:
    k = max(1, round(len(items) * ratio))
    return rng.sample(items, min(k, len(items)))


def build_yolo_toy(src: Path, output: Path, ratio: float, rng: random.Random, dry_run: bool) -> dict:
    counters = {}
    for split in SPLITS:
        images_dir = src / "images" / split
        if not images_dir.is_dir():
            continue
        images = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        selected = sample(images, ratio, rng)
        counters[split] = len(selected)
        if dry_run:
            continue
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in selected:
            shutil.copy2(img, output / "images" / split / img.name)
            label = src / "labels" / split / (img.stem + ".txt")
            dst_label = output / "labels" / split / (img.stem + ".txt")
            if label.exists():
                shutil.copy2(label, dst_label)
            else:
                dst_label.touch()

    # Copy data.yaml
    data_yaml = src / "data.yaml"
    if not dry_run and data_yaml.exists():
        shutil.copy2(data_yaml, output / "data.yaml")

    return counters


def build_seq_toy(src: Path, output: Path, ratio: float, rng: random.Random, dry_run: bool) -> dict:
    counters = {}
    for split in SPLITS:
        split_dir = src / split
        if not split_dir.is_dir():
            continue
        total_selected = 0
        for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            sequences = sorted(d for d in class_dir.iterdir() if d.is_dir())
            selected = sample(sequences, ratio, rng)
            total_selected += len(selected)
            if dry_run:
                continue
            for seq in selected:
                dst = output / split / class_dir.name / seq.name
                dst.mkdir(parents=True, exist_ok=True)
                for subdir in ("images", "labels"):
                    src_sub = seq / subdir
                    if src_sub.is_dir():
                        shutil.copytree(src_sub, dst / subdir, dirs_exist_ok=True)
        counters[split] = total_selected

    return counters


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    yolo_src: Path = args["yolo_src"]
    seq_src: Path = args["seq_src"]
    yolo_output: Path = args["yolo_output"]
    seq_output: Path = args["seq_output"]
    ratio: float = args["ratio"]
    dry_run: bool = args["dry_run"]
    rng = random.Random(args["random_seed"])

    if not dry_run:
        for out in (yolo_output, seq_output):
            if out.exists():
                shutil.rmtree(out)

    logging.info(f"Sampling {ratio*100:.0f}% from {yolo_src} → {yolo_output}")
    yolo_counts = build_yolo_toy(yolo_src, yolo_output, ratio, rng, dry_run)

    logging.info(f"Sampling {ratio*100:.0f}% from {seq_src} → {seq_output}")
    seq_counts = build_seq_toy(seq_src, seq_output, ratio, rng, dry_run)

    print(f"\n{'='*50}")
    print(f"{'DRY RUN — ' if dry_run else ''}Toy dataset ({ratio*100:.0f}%)")
    print("  YOLO images:")
    for split in SPLITS:
        print(f"    {split:<6}: {yolo_counts.get(split, 0):>5}")
    print("  Sequential sequences:")
    for split in SPLITS:
        print(f"    {split:<6}: {seq_counts.get(split, 0):>5}")
    print(f"{'='*50}\n")

    if dry_run:
        print("Dry run — nothing written.")
