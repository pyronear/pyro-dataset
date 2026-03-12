"""
Build a sequential dataset from wildfire (wf) and false-positive (fp) sequences.

Each sequence is copied as a folder (images/ + labels/ preserved as-is).
FP sequences are balanced to 50% per split at the sequence level.

Output structure:
    <output-train-val>/
      train/
        <wf_sequence>/
          images/
          labels/
        <fp_sequence>/
          ...
      val/
        ...

    <output-test>/
      test/
        ...

Sampling strategy for FP (by max sequence score):
  Pick the top-N FP sequences by max detection score, with random shuffle
  for tie-breaking.

Usage:
    python build_sequential_dataset.py
    python build_sequential_dataset.py --dry-run

Arguments:
    --wf-registry       Path to wildfire registry.json (default: data/raw/wildfire/registry.json).
    --wf-data-dir       Path to wildfire sequence folders (default: data/raw/wildfire/data).
    --fp-registry       Path to fp registry.json (default: data/raw/fp/registry.json).
    --fp-data-dir       Path to fp sequence folders (default: data/raw/fp/data).
    --output-train-val  Output directory for train+val (default: data/processed/sequential_train_val).
    --output-test       Output directory for test (default: data/processed/sequential_test).
    --random-seed       Random seed for tie-breaking (default: 0).
    --dry-run           Print plan without writing anything.
    -log, --loglevel    Logging level (default: info).
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path


SPLITS = ["train", "val", "test"]


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sequential dataset from wf and fp sequences."
    )
    parser.add_argument("--wf-registry", type=Path, default=Path("data/raw/wildfire/registry.json"))
    parser.add_argument("--wf-data-dir", type=Path, default=Path("data/raw/wildfire/data"))
    parser.add_argument("--fp-registry", type=Path, default=Path("data/raw/fp/registry.json"))
    parser.add_argument("--fp-data-dir", type=Path, default=Path("data/raw/fp/data"))
    parser.add_argument("--output-train-val", type=Path, default=Path("data/processed/sequential_train_val"))
    parser.add_argument("--output-test", type=Path, default=Path("data/processed/sequential_test"))
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("-log", "--loglevel", default="info")
    return parser


def load_registry(registry_path: Path) -> list[dict]:
    with registry_path.open() as f:
        return json.load(f)["sequences"]


def seq_max_score(seq_path: Path) -> float:
    """Return the max detection score across all label files in a sequence."""
    labels_dir = seq_path / "labels"
    if not labels_dir.is_dir():
        return 0.0
    best = 0.0
    for label_file in labels_dir.iterdir():
        if label_file.suffix != ".txt":
            continue
        try:
            for line in label_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    best = max(best, float(parts[5]))
        except Exception:
            pass
    return best


def sample_fp_sequences(
    sequences: list[tuple[float, Path]],
    quota: int,
    rng: random.Random,
) -> list[Path]:
    """Pick top-N FP sequences by max detection score, random shuffle for ties."""
    # Shuffle first for tie-breaking, then stable sort by score descending
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    indices.sort(key=lambda i: -sequences[i][0])
    return [sequences[i][1] for i in indices[:quota]]


def copy_sequence(src: Path, dst: Path, dry_run: bool) -> None:
    """Copy a sequence folder (images/ + labels/) to dst."""
    if dry_run:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for subdir in ("images", "labels"):
        src_sub = src / subdir
        dst_sub = dst / subdir
        if src_sub.is_dir():
            shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    wf_registry_path: Path = args["wf_registry"]
    wf_data_dir: Path = args["wf_data_dir"]
    fp_registry_path: Path = args["fp_registry"]
    fp_data_dir: Path = args["fp_data_dir"]
    output_train_val: Path = args["output_train_val"]
    output_test: Path = args["output_test"]
    dry_run: bool = args["dry_run"]
    rng = random.Random(args["random_seed"])

    if not dry_run:
        for out in (output_train_val, output_test):
            if out.exists():
                shutil.rmtree(out)

    wf_sequences = load_registry(wf_registry_path)
    fp_sequences = load_registry(fp_registry_path)
    logging.info(f"Loaded {len(wf_sequences)} WF sequences, {len(fp_sequences)} FP sequences")

    # Group WF sequences by split
    wf_by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for seq in wf_sequences:
        wf_by_split[seq["split"]].append(seq)

    # Group FP sequences by split and score them
    fp_by_split: dict[str, list[tuple[float, Path]]] = {s: [] for s in SPLITS}
    fp_missing = 0
    for seq in fp_sequences:
        split = seq["split"]
        seq_path = fp_data_dir / seq["folder"]
        if not seq_path.is_dir():
            logging.warning(f"FP sequence folder not found: {seq_path}")
            fp_missing += 1
            continue
        score = seq_max_score(seq_path)
        fp_by_split[split].append((score, seq_path))

    # Map split → output root
    split_output = {
        "train": output_train_val,
        "val": output_train_val,
        "test": output_test,
    }

    # Process each split
    counters: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        wf_seqs = wf_by_split[split]
        n_wf = len(wf_seqs)
        quota = n_wf  # 50/50 balance at sequence level
        out = split_output[split]

        selected_fp = sample_fp_sequences(fp_by_split[split], quota, rng)
        n_fp = len(selected_fp)

        logging.info(f"{split}: {n_wf} WF + {n_fp} FP sequences → {out}/{split}/")

        wf_missing = 0
        for seq in wf_seqs:
            src = wf_data_dir / seq["folder"]
            if not src.is_dir():
                logging.warning(f"WF sequence folder not found: {src}")
                wf_missing += 1
                continue
            copy_sequence(src, out / split / seq["folder"], dry_run)

        for seq_path in selected_fp:
            copy_sequence(seq_path, out / split / seq_path.name, dry_run)

        counters[split] = {"wf": n_wf - wf_missing, "fp": n_fp}

    print(f"\n{'='*55}")
    print(f"{'DRY RUN — ' if dry_run else ''}Sequential dataset")
    for split in SPLITS:
        wf = counters[split]["wf"]
        fp = counters[split]["fp"]
        total = wf + fp
        ratio = fp / total * 100 if total else 0
        print(f"  {split:<6}: {wf:>5} WF + {fp:>5} FP = {total:>5} sequences  ({ratio:.0f}% FP)")
    if fp_missing:
        print(f"  missing FP folders: {fp_missing}")
    print(f"{'='*55}\n")

    if dry_run:
        print("Dry run — nothing written.")
