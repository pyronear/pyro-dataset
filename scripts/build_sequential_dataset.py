"""
Build a sequential dataset from wildfire (wf) and false-positive (fp) sequences.

Each sequence is copied as a folder (images/ + labels/ preserved as-is).
FP sequences are balanced to 50% per split at the sequence level.

Output structure:
    <output-train-val>/
      train/
        wildfire/
          <wf_sequence>/
            images/
            labels/
        fp/
          <fp_sequence>/
            ...
      val/
        ...

    <output-test>/
      test/
        wildfire/
          ...
        fp/
          ...

Sampling strategy for FP (train and val): two-stage clustering
  Stage 1 — bbox-overlap atoms per camera (NMS top-1 main bbox; intra-camera
            union-find with IoU > match_iou).
  Stage 2 — KMeans(k = quota) on the DINOv2 embedding of each atom rep.
            Per cluster, pick the atom rep closest to the cluster centroid.
  Requires per-split DINOv2 embeddings produced by embed_fp_for_selection.py.

  Test negatives are NOT selected here: they come verbatim from the
  append-only lockfile (--test-lockfile), maintained by
  scripts/freeze_test_selection.py so the test set only ever grows.

Usage:
    python build_sequential_dataset.py
    python build_sequential_dataset.py --dry-run

Arguments:
    --wf-registry       Path to wildfire registry.json (default: data/raw/wildfire/registry.json).
    --wf-data-dir       Path to wildfire sequence folders (default: data/raw/wildfire/data).
    --fp-registry       Path to fp registry.json (default: data/raw/fp/registry.json).
    --fp-data-dir       Path to fp sequence folders (default: data/raw/fp/data).
    --embeddings-dir    Per-split DINOv2 embeddings root (default: data/interim/fp_sequence_embeddings).
    --output-train-val  Output directory for train+val (default: data/processed/sequential_train_val).
    --output-test       Output directory for test (default: data/processed/sequential_test).
    --test-lockfile     Append-only lockfile holding the test FP folders
                        (default: data/raw/sequential_test_lock.json).
    --random-seed       Random seed for KMeans init (default: 0).
    --nms-iou           NMS IoU for intra-sequence main bbox (default: 0.3).
    --match-iou         IoU threshold for intra-camera atoms (default: 0.7).
    --dry-run           Print plan without writing anything.
    -log, --loglevel    Logging level (default: info).
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from pyro_dataset.fp.lockfile import load_lockfile, validate_lockfile
from pyro_dataset.fp.selection import (
    load_embeddings,
    partition_pinned,
    remaining_quota,
    two_stage_select,
)

SPLITS = ["train", "val", "test"]


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sequential dataset from wf and fp sequences."
    )
    parser.add_argument(
        "--wf-registry", type=Path, default=Path("data/raw/wildfire/registry.json")
    )
    parser.add_argument(
        "--wf-data-dir", type=Path, default=Path("data/raw/wildfire/data")
    )
    parser.add_argument(
        "--fp-registry", type=Path, default=Path("data/raw/fp/registry.json")
    )
    parser.add_argument("--fp-data-dir", type=Path, default=Path("data/raw/fp/data"))
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/interim/fp_sequence_embeddings"),
        help="Per-split DINOv2 embeddings root (read for the two-stage FP selection).",
    )
    parser.add_argument(
        "--output-train-val",
        type=Path,
        default=Path("data/processed/sequential_train_val"),
    )
    parser.add_argument(
        "--output-test", type=Path, default=Path("data/processed/sequential_test")
    )
    parser.add_argument(
        "--test-lockfile",
        type=Path,
        default=Path("data/raw/sequential_test_lock.json"),
        help="Append-only lockfile that IS the FP half of the test split.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--match-iou", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("-log", "--loglevel", default="info")
    return parser


def load_registry(registry_path: Path) -> list[dict]:
    with registry_path.open() as f:
        return json.load(f)["sequences"]


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


def frozen_test_selection(
    lockfile_path: Path, quota: int, registered: set[str], data_dir: Path
) -> list[Path]:
    """The test FP folders, exactly as frozen — the build selects nothing.

    Errors, never warns: a test set that cannot be materialised exactly as
    the lockfile records breaks comparability across releases, which is the
    lockfile's whole purpose. A quota mismatch usually means an import grew
    the test WF pool and `scripts/freeze_test_selection.py` was not rerun.
    """
    if not lockfile_path.is_file():
        raise SystemExit(
            f"missing test lockfile {lockfile_path} — "
            "run scripts/freeze_test_selection.py"
        )
    folders = load_lockfile(lockfile_path)
    if len(folders) != quota:
        raise SystemExit(
            f"test lockfile has {len(folders)} folders but the quota is {quota} "
            "— run scripts/freeze_test_selection.py (a persistent shortfall "
            "means the test FP pool is exhausted)"
        )
    problems = validate_lockfile(folders, registered, data_dir)
    if problems:
        raise SystemExit("test lockfile invalid:\n  " + "\n  ".join(problems))
    return [data_dir / folder for folder in folders]


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())

    wf_registry_path: Path = args["wf_registry"]
    wf_data_dir: Path = args["wf_data_dir"]
    fp_registry_path: Path = args["fp_registry"]
    fp_data_dir: Path = args["fp_data_dir"]
    embeddings_dir: Path = args["embeddings_dir"]
    output_train_val: Path = args["output_train_val"]
    output_test: Path = args["output_test"]
    test_lockfile: Path = args["test_lockfile"]
    dry_run: bool = args["dry_run"]
    nms_iou: float = args["nms_iou"]
    match_iou: float = args["match_iou"]
    seed: int = args["random_seed"]

    if not dry_run:
        for out in (output_train_val, output_test):
            if out.exists():
                shutil.rmtree(out)

    wf_sequences = load_registry(wf_registry_path)
    fp_sequences = load_registry(fp_registry_path)
    logging.info(
        f"Loaded {len(wf_sequences)} WF sequences, {len(fp_sequences)} FP sequences"
    )

    # Group WF sequences by split
    wf_by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for seq in wf_sequences:
        wf_by_split[seq["split"]].append(seq)

    # Map split → output root
    split_output = {
        "train": output_train_val,
        "val": output_train_val,
        "test": output_test,
    }

    fp_missing_total = 0
    counters: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        wf_seqs = wf_by_split[split]
        n_wf = len(wf_seqs)
        quota = n_wf  # 50/50 balance at sequence level
        out = split_output[split]

        if split == "test":
            # Test negatives are frozen: the lockfile is the selection
            # (docs/specs/2026-08-14-annotator-test-growth-design.md §3).
            selected_fp_paths = frozen_test_selection(
                test_lockfile,
                quota,
                {s["folder"] for s in fp_sequences if s["split"] == "test"},
                fp_data_dir,
            )
            n_fp = len(selected_fp_paths)
            strategy = "frozen lockfile"
        else:
            # Annotator-sourced FP sequences were chosen deliberately upstream, so
            # they are included first and the clustering fills only what is left.
            pinned_seqs, _pool_seqs = partition_pinned(
                [s for s in fp_sequences if s["split"] == split]
            )
            # Registered but absent from disk: warn like the WF path does rather
            # than copying an empty folder into the dataset. Not counted as missing
            # here — the embeddings loop below counts the same folder.
            for seq in list(pinned_seqs):
                if not (fp_data_dir / seq["folder"]).is_dir():
                    logging.warning(
                        f"pinned FP sequence folder not found: {fp_data_dir / seq['folder']}"
                    )
                    pinned_seqs.remove(seq)
            pinned_paths = [fp_data_dir / s["folder"] for s in pinned_seqs]
            quota = remaining_quota(quota, pinned_seqs)

            # Load DINOv2 embeddings + per-sequence metadata for this split. The
            # annotator cohort is absent by construction — embed_fp_for_selection
            # skips it — so only folders missing from disk are dropped here.
            emb, items = load_embeddings(embeddings_dir, split)
            keep_mask = []
            items_kept: list[dict] = []
            for it in items:
                ok = (fp_data_dir / it["sequence_folder"]).is_dir()
                keep_mask.append(ok)
                if not ok:
                    fp_missing_total += 1
                if ok:
                    items_kept.append(it)
            if not all(keep_mask):
                import numpy as _np

                emb = emb[_np.asarray(keep_mask)]

            selected_idx = two_stage_select(
                items=items_kept,
                embeddings=emb,
                data_dir=fp_data_dir,
                quota=quota,
                nms_iou=nms_iou,
                match_iou=match_iou,
                seed=seed,
            )
            selected_fp_paths = pinned_paths + [
                fp_data_dir / items_kept[i]["sequence_folder"] for i in selected_idx
            ]
            n_fp = len(selected_fp_paths)
            strategy = f"two_stage; {len(pinned_seqs)} pinned from the annotator"

        logging.info(
            f"{split}: {n_wf} WF + {n_fp} FP sequences → {out}/{split}/  ({strategy})"
        )

        wf_missing = 0
        for seq in wf_seqs:
            src = wf_data_dir / seq["folder"]
            if not src.is_dir():
                logging.warning(f"WF sequence folder not found: {src}")
                wf_missing += 1
                continue
            copy_sequence(src, out / split / "wildfire" / seq["folder"], dry_run)

        for seq_path in selected_fp_paths:
            copy_sequence(seq_path, out / split / "fp" / seq_path.name, dry_run)

        counters[split] = {"wf": n_wf - wf_missing, "fp": n_fp}
    fp_missing = fp_missing_total

    print(f"\n{'=' * 55}")
    print(f"{'DRY RUN — ' if dry_run else ''}Sequential dataset")
    for split in SPLITS:
        wf = counters[split]["wf"]
        fp = counters[split]["fp"]
        total = wf + fp
        ratio = fp / total * 100 if total else 0
        print(
            f"  {split:<6}: {wf:>5} WF + {fp:>5} FP = {total:>5} sequences  ({ratio:.0f}% FP)"
        )
    if fp_missing:
        print(f"  missing FP folders: {fp_missing}")
    print(f"{'=' * 55}\n")

    if dry_run:
        print("Dry run — nothing written.")
