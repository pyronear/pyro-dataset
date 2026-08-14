"""
CLI freezing the FP half of `sequential_test` into an append-only lockfile.

Manual, never a DVC stage, for the same reason as plan_annotator_import.py:
it reads and rewrites accumulated state, and a stage would regenerate that
state from deps, destroying the pinning it exists to provide. The build
(`build_sequential_dataset.py`) copies the lockfile verbatim for test and
performs no selection — so every release's test set is a byte-identical
superset of the previous one, and models stay comparable across releases.

Each run: quota = number of test WF sequences in the registry; the lockfile
is loaded (or, once, bootstrapped from the built dataset — never recomputed);
annotator-sourced test FPs are pinned up to the quota in registry order;
remaining slots are filled by the existing two-stage selection over the test
pool minus what is already frozen. Entries are never removed or reordered.

See docs/specs/2026-08-14-annotator-test-growth-design.md.

Usage:
    python scripts/freeze_test_selection.py
    python scripts/freeze_test_selection.py --dry-run

Arguments:
    --wf-registry      Wildfire registry (default: data/raw/wildfire/registry.json).
    --fp-registry      FP registry (default: data/raw/fp/registry.json).
    --fp-data-dir      FP sequence folders (default: data/raw/fp/data).
    --embeddings-dir   Per-split DINOv2 embeddings root (default: data/interim/fp_sequence_embeddings).
    --lockfile         The lockfile (default: data/raw/sequential_test_lock.json).
    --bootstrap-from   Built sequential test dir to seed a missing lockfile from
                       (default: data/processed/sequential_test).
    --nms-iou          NMS IoU for the fill selection (default: 0.3).
    --match-iou        Atom IoU for the fill selection (default: 0.7).
    --random-seed      Seed for the fill selection (default: 0).
    --dry-run          Print the outcome without writing.
    -log, --loglevel   Logging level (default: info).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pyro_dataset.fp.lockfile import (
    extend_with_pins,
    load_lockfile,
    save_lockfile,
    validate_lockfile,
)
from pyro_dataset.fp.selection import load_embeddings, two_stage_select
from pyro_dataset.ingest import ANNOTATOR_SOURCE


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the FP half of sequential_test into the lockfile."
    )
    parser.add_argument(
        "--wf-registry", type=Path, default=Path("data/raw/wildfire/registry.json")
    )
    parser.add_argument(
        "--fp-registry", type=Path, default=Path("data/raw/fp/registry.json")
    )
    parser.add_argument("--fp-data-dir", type=Path, default=Path("data/raw/fp/data"))
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/interim/fp_sequence_embeddings"),
    )
    parser.add_argument(
        "--lockfile", type=Path, default=Path("data/raw/sequential_test_lock.json")
    )
    parser.add_argument(
        "--bootstrap-from",
        type=Path,
        default=Path("data/processed/sequential_test"),
        help="Built test dataset to seed a missing lockfile from, exactly.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--match-iou", type=float, default=0.7)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser


def load_registry(path: Path) -> list[dict]:
    return json.loads(path.read_text())["sequences"]


def bootstrap_folders(built_test: Path) -> list[str]:
    """Seed a first lockfile from the built dataset, never by recomputing.

    Recomputing would trust the two-stage selection to reproduce what shipped
    bit-identically — the exact assumption the lockfile exists to retire. The
    built dataset is what current models were evaluated on, so it is the truth.
    """
    fp_dir = built_test / "test" / "fp"
    if not fp_dir.is_dir():
        raise SystemExit(
            f"no lockfile and no built test set at {fp_dir} — "
            "run `dvc pull` or `dvc repro build_sequential_dataset` first"
        )
    return sorted(d.name for d in fp_dir.iterdir() if d.is_dir())


def main() -> None:
    args = vars(make_cli_parser().parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    lockfile_path: Path = args["lockfile"]
    data_dir: Path = args["fp_data_dir"]
    dry_run: bool = args["dry_run"]

    wf_sequences = load_registry(args["wf_registry"])
    fp_sequences = load_registry(args["fp_registry"])
    quota = sum(1 for s in wf_sequences if s["split"] == "test")
    test_fp = [s for s in fp_sequences if s["split"] == "test"]
    registered = {s["folder"] for s in test_fp}

    bootstrapped = not lockfile_path.is_file()
    frozen = (
        bootstrap_folders(args["bootstrap_from"])
        if bootstrapped
        else load_lockfile(lockfile_path)
    )

    pins = [s["folder"] for s in test_fp if s.get("source") == ANNOTATOR_SOURCE]
    try:
        folders = extend_with_pins(frozen, pins, quota)
    except ValueError as error:
        raise SystemExit(str(error))
    n_pinned = len(folders) - len(frozen)
    deferred = [f for f in pins if f not in set(folders)]

    n_filled = 0
    if len(folders) < quota:
        emb, items = load_embeddings(args["embeddings_dir"], "test")
        taken = set(folders)
        keep = [
            i
            for i, item in enumerate(items)
            if item["sequence_folder"] not in taken
            and (data_dir / item["sequence_folder"]).is_dir()
        ]
        items_kept = [items[i] for i in keep]
        selected = two_stage_select(
            items=items_kept,
            embeddings=emb[np.asarray(keep, dtype=int)],
            data_dir=data_dir,
            quota=quota - len(folders),
            nms_iou=args["nms_iou"],
            match_iou=args["match_iou"],
            seed=args["random_seed"],
        )
        fill = [items_kept[i]["sequence_folder"] for i in selected]
        folders += fill
        n_filled = len(fill)

    problems = validate_lockfile(folders, registered, data_dir)
    if problems:
        for problem in problems:
            logging.error(problem)
        raise SystemExit(f"{len(problems)} problem(s) — lockfile not written")

    print(f"\n{'DRY RUN — ' if dry_run else ''}Test selection freeze")
    print(f"  quota    : {quota}")
    print(f"  frozen   : {len(frozen)}{'  (bootstrapped)' if bootstrapped else ''}")
    print(f"  pinned   : +{n_pinned} annotator ({len(deferred)} deferred beyond quota)")
    print(f"  filled   : +{n_filled} by two-stage selection")
    if len(folders) < quota:
        logging.warning(
            f"only {len(folders)} of {quota} slots filled — the test FP pool "
            "is exhausted; the build will refuse until more test FPs exist"
        )
    if dry_run:
        print("Dry run — nothing written.")
        return
    save_lockfile(lockfile_path, folders)
    print(f"  lockfile : {lockfile_path}")
    print("\nCommit the lockfile together with the registries it froze against.")


if __name__ == "__main__":
    sys.exit(main())
