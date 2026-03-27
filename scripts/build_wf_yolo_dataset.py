"""
Build a YOLO dataset from wildfire sequences registered in registry.json.

For each sequence, selects up to --max-per-seq images that have a non-empty
label file, chosen at random. Copies images and labels into the standard
YOLO folder structure under the output directory.

Output structure:
    <output>/
      images/train/
      images/val/
      images/test/
      labels/train/
      labels/val/
      labels/test/

Usage:
    python build_wf_yolo_dataset.py
    python build_wf_yolo_dataset.py --max-per-seq 3
    python build_wf_yolo_dataset.py --dry-run

Arguments:
    --registry      Path to registry.json (default: data/raw/wildfire/registry.json).
    --data-dir      Path to sequence folders (default: data/raw/wildfire/data).
    --output        Output directory (default: data/processed/wildfire_yolo).
    --max-per-seq   Max images per sequence (default: 5).
    --random-seed   Random seed (default: 0).
    --dry-run       Print what would be copied without writing anything.
    -log, --loglevel  Logging level (default: info).
"""

import argparse
import logging
import random
import shutil
from pathlib import Path


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset from wildfire sequences."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/raw/wildfire/registry.json"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/wildfire/data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wildfire_yolo"),
    )
    parser.add_argument("--max-per-seq", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("-log", "--loglevel", default="info")
    return parser


def load_registry(registry_path: Path) -> list[dict]:
    import json
    with registry_path.open() as f:
        return json.load(f)["sequences"]


def labeled_images(seq_path: Path) -> list[Path]:
    """Return image paths that have a non-empty label file."""
    images_dir = seq_path / "images"
    labels_dir = seq_path / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []
    result = []
    for img in images_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = labels_dir / (img.stem + ".txt")
        if label.exists() and label.stat().st_size > 0:
            result.append(img)
    return sorted(result)


def copy_label_normalized(src: Path, dst: Path) -> None:
    """Copy a YOLO label file keeping only the 5 required columns (class x y w h).

    Raw label files may contain a 6th confidence column from model predictions.
    YOLO training requires exactly 5 columns, so any extra columns are stripped.
    """
    lines = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            lines.append(" ".join(parts[:5]))
    dst.write_text("\n".join(lines))


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    registry_path: Path = args["registry"]
    data_dir: Path = args["data_dir"]
    output: Path = args["output"]
    max_per_seq: int = args["max_per_seq"]
    dry_run: bool = args["dry_run"]
    rng = random.Random(args["random_seed"])

    sequences = load_registry(registry_path)
    logging.info(f"Loaded {len(sequences)} sequences from registry")

    splits = ["train", "val", "test"]
    counters = {s: 0 for s in splits}
    skipped = 0

    if not dry_run:
        if output.exists():
            shutil.rmtree(output)
        for split in splits:
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        folder = seq["folder"]
        split = seq["split"]
        seq_path = data_dir / folder

        if not seq_path.is_dir():
            logging.warning(f"Sequence folder not found: {seq_path}")
            skipped += 1
            continue

        candidates = labeled_images(seq_path)
        if not candidates:
            logging.debug(f"No labeled images in {folder}")
            skipped += 1
            continue

        selected = rng.sample(candidates, min(max_per_seq, len(candidates)))

        for img in selected:
            label = seq_path / "labels" / (img.stem + ".txt")
            dst_img = output / "images" / split / img.name
            dst_label = output / "labels" / split / label.name
            logging.debug(f"  {split}: {img.name}")
            if not dry_run:
                shutil.copy2(img, dst_img)
                copy_label_normalized(label, dst_label)
            counters[split] += 1

    total = sum(counters.values())
    print(f"\n{'='*50}")
    print(f"{'DRY RUN — ' if dry_run else ''}Images copied: {total}")
    for split in splits:
        print(f"  {split:<6}: {counters[split]}")
    print(f"  skipped sequences: {skipped}")
    print(f"{'='*50}\n")

    if dry_run:
        print("Dry run — nothing written.")
