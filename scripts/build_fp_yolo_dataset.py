"""
Build a FP (false positive) YOLO dataset from fp sequences registered in registry.json.

FP images are background images (empty labels). Their label files contain model
predictions (class x y w h score) used only for scoring/ranking — the output label
files are empty (no smoke annotation).

Quota per split (in IMAGES, computed from wildfire image counts):
  train / val : FP = 10% of total  →  n_fp = n_wf / 9
  test        : FP = 50% of total  →  n_fp = n_wf

Sampling strategy:
  • train, val — TWO-STAGE clustering (selects diverse sequences):
      1. Bbox-overlap grouping per camera (NMS top-1 main bbox; intra-camera
         union-find with IoU > 0.7) → "atoms" (each ≡ same recurring artefact
         on the same camera).
      2. KMeans(k = quota) on the DINOv2 embedding of each atom's representative
         (closest to the atom's embedding centroid). For each KMeans cluster,
         pick the atom whose rep is closest to the cluster centroid; output the
         highest-scoring frame of that rep sequence as the FP image.
      Requires per-split DINOv2 embeddings produced by embed_fp_for_selection.py.

  • test — round-robin by max score (legacy). Each round, from every sequence
      pick the unselected image with the highest max detection score. Used
      because test quota typically exceeds the number of FP sequences, making
      diversity-based selection moot.

Output structure:
    <output>/
      images/train/
      images/val/
      images/test/
      labels/train/    ← empty .txt files (background)
      labels/val/
      labels/test/

Usage:
    python build_fp_yolo_dataset.py
    python build_fp_yolo_dataset.py --dry-run

Arguments:
    --registry         Path to fp registry.json (default: data/raw/fp/registry.json).
    --data-dir         Path to fp sequence folders (default: data/raw/fp/data).
    --wf-dataset       Path to wildfire_yolo dataset to read WF counts from
                       (default: data/processed/wildfire_yolo).
    --embeddings-dir   Per-split DINOv2 embeddings root for train/val
                       (default: data/interim/fp_sequence_embeddings).
    --output           Output directory (default: data/processed/fp_yolo).
    --random-seed      Random seed for KMeans init + round-robin tie-break (default: 0).
    --nms-iou          NMS IoU for intra-sequence main bbox extraction (default: 0.3).
    --match-iou        IoU threshold for intra-camera bbox-overlap atoms (default: 0.7).
    --dry-run          Print plan without writing anything.
    -log, --loglevel   Logging level (default: info).
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

from pyro_dataset.fp.selection import load_embeddings, two_stage_select


SPLITS = ["train", "val", "test"]
FP_RATIO = {"train": 0.1, "val": 0.1, "test": 0.5}
TWO_STAGE_SPLITS = {"train", "val"}  # test stays on round-robin


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a FP YOLO dataset from fp sequences."
    )
    parser.add_argument("--registry", type=Path, default=Path("data/raw/fp/registry.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fp/data"))
    parser.add_argument("--wf-dataset", type=Path, default=Path("data/processed/wildfire_yolo"))
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/interim/fp_sequence_embeddings"),
        help="Per-split DINOv2 embeddings root (read for train/val two-stage selection).",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/fp_yolo"))
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--nms-iou", type=float, default=0.3,
                        help="NMS IoU threshold for intra-sequence main bbox.")
    parser.add_argument("--match-iou", type=float, default=0.7,
                        help="IoU threshold for intra-camera bbox-overlap grouping.")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("-log", "--loglevel", default="info")
    return parser


def load_registry(registry_path: Path) -> list[dict]:
    with registry_path.open() as f:
        return json.load(f)["sequences"]


def count_wf_images(wf_dataset: Path) -> dict[str, int]:
    counts = {}
    for split in SPLITS:
        images_dir = wf_dataset / "images" / split
        if images_dir.is_dir():
            counts[split] = sum(
                1 for f in images_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        else:
            counts[split] = 0
    return counts


def max_score(label_path: Path) -> float:
    """Return the max detection score from a label file (6th column)."""
    best = 0.0
    try:
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 6:
                best = max(best, float(parts[5]))
    except Exception:
        pass
    return best


def scored_images(seq_path: Path) -> list[tuple[float, Path]]:
    """Return (score, image_path) pairs for all images with non-empty labels."""
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
            result.append((max_score(label), img))
    return result


def round_robin_sample(
    sequences: list[list[tuple[float, Path]]],
    quota: int,
    rng: random.Random,
) -> list[Path]:
    """
    Round-robin selection: each round pick the highest-score unselected image
    from each sequence. Sequences are shuffled once for tie-breaking order.

    sequences: list of candidate lists per sequence, each pre-sorted descending by score.
    """
    pointers = [0] * len(sequences)
    order = list(range(len(sequences)))
    rng.shuffle(order)
    selected = []

    while len(selected) < quota:
        progress = False
        for i in order:
            if len(selected) >= quota:
                break
            if pointers[i] < len(sequences[i]):
                _, img = sequences[i][pointers[i]]
                selected.append(img)
                pointers[i] += 1
                progress = True
        if not progress:
            break  # all sequences exhausted

    return selected


# Two-stage helpers live in src/pyro_dataset/fp/selection.py and are imported above.


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    registry_path: Path = args["registry"]
    data_dir: Path = args["data_dir"]
    wf_dataset: Path = args["wf_dataset"]
    embeddings_dir: Path = args["embeddings_dir"]
    output: Path = args["output"]
    dry_run: bool = args["dry_run"]
    nms_iou: float = args["nms_iou"]
    match_iou: float = args["match_iou"]
    seed: int = args["random_seed"]
    rng = random.Random(seed)

    sequences = load_registry(registry_path)
    logging.info(f"Loaded {len(sequences)} FP sequences from registry")

    wf_counts = count_wf_images(wf_dataset)
    quotas = {
        split: int(wf_counts[split] * FP_RATIO[split] / (1 - FP_RATIO[split]))
        for split in SPLITS
    }
    logging.info(f"WF counts: {wf_counts}")
    logging.info(f"FP quotas: {quotas}")

    # Group sequences by split for round-robin (test) and metadata lookups
    by_split_seqs: dict[str, list[dict]] = {s: [] for s in SPLITS}
    by_split_candidates: dict[str, list[list[tuple[float, Path]]]] = {s: [] for s in SPLITS}
    missing = 0
    for seq in sequences:
        folder = seq["folder"]
        split = seq["split"]
        seq_path = data_dir / folder
        if not seq_path.is_dir():
            logging.warning(f"Sequence folder not found: {seq_path}")
            missing += 1
            continue
        by_split_seqs[split].append(seq)
        candidates = scored_images(seq_path)
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            by_split_candidates[split].append(candidates)

    if not dry_run:
        if output.exists():
            shutil.rmtree(output)
        for split in SPLITS:
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    strategies: dict[str, str] = {}
    for split in SPLITS:
        quota = quotas[split]
        if split in TWO_STAGE_SPLITS:
            emb, items = load_embeddings(embeddings_dir, split)
            selected_idx = two_stage_select(
                items=items,
                embeddings=emb,
                data_dir=data_dir,
                quota=quota,
                nms_iou=nms_iou,
                match_iou=match_iou,
                seed=seed,
            )
            selected = [
                data_dir / items[i]["sequence_folder"] / "images" / items[i]["image_name"]
                for i in selected_idx
            ]
            strategies[split] = "two_stage"
        else:
            selected = round_robin_sample(by_split_candidates[split], quota, rng)
            strategies[split] = "round_robin"
        counters[split] = len(selected)
        for img in selected:
            dst_img = output / "images" / split / img.name
            dst_label = output / "labels" / split / (img.stem + ".txt")
            logging.debug(f"  {split}: {img.name}")
            if not dry_run:
                shutil.copy2(img, dst_img)
                dst_label.touch()  # empty label = background

    total = sum(counters.values())
    print(f"\n{'='*60}")
    print(f"{'DRY RUN — ' if dry_run else ''}FP images: {total}")
    for split in SPLITS:
        ratio = FP_RATIO[split] * 100
        print(
            f"  {split:<6}: {counters[split]:>5}  "
            f"(target {ratio:.0f}% FP, quota {quotas[split]}, strategy={strategies[split]})"
        )
    if missing:
        print(f"  missing sequence folders: {missing}")
    print(f"{'='*60}\n")

    if dry_run:
        print("Dry run — nothing written.")
